#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp> 
#include <iostream>
#include <fstream>
#include <deque>
#include <map>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <chrono>

using namespace cv;
using namespace std;

// ---------------- Parameters ----------------
const string VIDEO_PATH = "C:\\Users\\kasat\\Downloads\\7652_Sunset_Sundown_1920x1080.mp4";
const string OUT_PATH = "C:\\Users\\kasat\\Desktop\\dip_vlom\\output_annotated1.mp4";
const string OUT_CSV = "C:\\Users\\kasat\\Desktop\\dip_vlom\\detections1.csv";

const double UPSCALE = 1.0;
const int MOG_HISTORY = 30;
const double MOG_VAR_THRESHOLD = 16;
const bool MOG_DETECT_SHADOWS = false;

const double MIN_AREA = 10;
const double MAX_AREA = 2000;
const int N_CONFIRM = 10;
const int MAX_MISSES = 6;
const double DIST_THRESH = 60.0;

// Morphology kernels
Mat KERNEL_OPEN = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
Mat KERNEL_CLOSE = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));

// Colors
Scalar COLOR_PENDING(0, 255, 255);
Scalar COLOR_CONFIRMED(0, 0, 255);
Scalar COLOR_TRACK(255, 0, 0);

// ---------------- Minimal centroid tracker ----------------
struct Track {
    int id;
    Point2f centroid;
    Rect bbox;
    int hits;
    int misses;
    deque<Point2f> history;
    bool confirmed;

    Track() : id(0), centroid(0, 0), bbox(0, 0, 0, 0), hits(0), misses(0), confirmed(false) {}
    Track(int tid, Point2f c, Rect b) : id(tid), centroid(c), bbox(b), hits(1), misses(0), confirmed(false) {
        history.push_back(c);
    }
};

class SimpleTracker {
private:
    int next_id;
    map<int, Track> tracks;
    double dist_threshold;
    int max_misses;
    int confirm_hits;

    double dist(Point2f a, Point2f b) {
        return hypot(a.x - b.x, a.y - b.y);
    }

public:
    SimpleTracker(double dt = DIST_THRESH, int mm = MAX_MISSES, int ch = N_CONFIRM)
        : next_id(1), dist_threshold(dt), max_misses(mm), confirm_hits(ch) {
    }

    void update(vector<pair<Point2f, Rect>>& detections) {
        set<int> assigned;
        vector<int> unmatched_tracks;

        // Build list of track ids
        vector<int> track_ids;
        for (auto& kv : tracks) track_ids.push_back(kv.first);
        unmatched_tracks = track_ids;

        // Greedy matching
        for (size_t det_idx = 0; det_idx < detections.size(); ++det_idx) {
            auto& det = detections[det_idx];
            Point2f c = det.first;
            Rect bbox = det.second;

            int best_tid = -1;
            double best_dist = 1e9;
            for (int tid : track_ids) {
                if (find(unmatched_tracks.begin(), unmatched_tracks.end(), tid) == unmatched_tracks.end())
                    continue;
                double d = dist(c, tracks[tid].centroid);
                if (d < best_dist) {
                    best_dist = d;
                    best_tid = tid;
                }
            }

            if (best_tid != -1 && best_dist <= dist_threshold) {
                Track& t = tracks[best_tid];
                t.centroid = c;
                t.bbox = bbox;
                t.history.push_back(c);
                if (t.history.size() > 50) t.history.pop_front();
                t.hits += 1;
                t.misses = 0;
                if (t.hits >= confirm_hits) t.confirmed = true;

                unmatched_tracks.erase(remove(unmatched_tracks.begin(), unmatched_tracks.end(), best_tid), unmatched_tracks.end());
                assigned.insert(det_idx);
            }
        }

        // Unmatched detections -> new tracks
        for (size_t det_idx = 0; det_idx < detections.size(); ++det_idx) {
            if (assigned.count(det_idx)) continue;
            auto& det = detections[det_idx];
            tracks[next_id] = Track(next_id, det.first, det.second);
            next_id++;
        }

        // Increase misses for unmatched tracks
        for (int tid : unmatched_tracks) {
            Track& t = tracks[tid];
            t.misses += 1;
            if (t.misses > max_misses) {
                tracks.erase(tid);
            }
        }
    }

    vector<Track> get_tracks() {
        vector<Track> res;
        for (auto& kv : tracks) res.push_back(kv.second);
        return res;
    }
};

// ---------------- Helper ----------------
Mat upscale_frame(const Mat& frame, double scale) {
    if (scale == 1.0) return frame;
    Mat out;
    resize(frame, out, Size(), scale, scale, INTER_LINEAR);
    return out;
}

// ---------------- Main processing ----------------
int main() {
    
    cout << "OpenCV threads: " << getNumThreads() << endl;
    cout << "OpenCV build info:\n";
    cout << getBuildInformation() << endl;

    setNumThreads(16);

    double total_gray = 0;
    double total_mog = 0;
    double total_thresh = 0;
    double total_open = 0;
    double total_close = 0;
    double total_contours = 0;
    double total_tracker = 0;
    double total_draw = 0;
    double total_write = 0;

    auto t_global_start = std::chrono::high_resolution_clock::now();

    VideoCapture cap(VIDEO_PATH);

    if (!cap.isOpened()) {
        cerr << "Cannot open video: " << VIDEO_PATH << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    if (fps == 0) fps = 25.0;

    int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    int w_s = (int)(w * UPSCALE);
    int h_s = (int)(h * UPSCALE);

    VideoWriter writer(OUT_PATH, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(w_s, h_s));

    Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorMOG2(MOG_HISTORY, MOG_VAR_THRESHOLD, MOG_DETECT_SHADOWS);

    SimpleTracker tracker;

    ofstream csv_file(OUT_CSV);
    csv_file << "frame_idx,track_id,x,y,w,h,confirmed\n";

    int frame_idx = 0;
    Mat frame;
    while (cap.read(frame)) {
        frame_idx++;

        Mat frame_up = upscale_frame(frame, UPSCALE);
        Mat gray;

        auto t1 = chrono::high_resolution_clock::now();
        cvtColor(frame_up, gray, COLOR_BGR2GRAY);
        auto t2 = chrono::high_resolution_clock::now();
        total_gray += chrono::duration<double, milli>(t2 - t1).count();

        Mat fgmask;
        t1 = chrono::high_resolution_clock::now();
        backSub->apply(gray, fgmask);
        t2 = chrono::high_resolution_clock::now();
        total_mog += chrono::duration<double, milli>(t2 - t1).count();

        t1 = chrono::high_resolution_clock::now();
        threshold(fgmask, fgmask, 127, 255, THRESH_BINARY);
        t2 = chrono::high_resolution_clock::now();
        total_thresh += chrono::duration<double, milli>(t2 - t1).count();

        t1 = chrono::high_resolution_clock::now();
        morphologyEx(fgmask, fgmask, MORPH_OPEN, KERNEL_OPEN);
        t2 = chrono::high_resolution_clock::now();
        total_open += chrono::duration<double, milli>(t2 - t1).count();

        t1 = chrono::high_resolution_clock::now();
        morphologyEx(fgmask, fgmask, MORPH_CLOSE, KERNEL_CLOSE);
        t2 = chrono::high_resolution_clock::now();
        total_close += chrono::duration<double, milli>(t2 - t1).count();


        vector<vector<Point>> contours;
        t1 = chrono::high_resolution_clock::now();
        findContours(fgmask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        t2 = chrono::high_resolution_clock::now();
        total_contours += chrono::duration<double, milli>(t2 - t1).count();

        vector<pair<Point2f, Rect>> detections;
        for (auto& cnt : contours) {
            double area = contourArea(cnt);
            if (area < MIN_AREA || area > MAX_AREA) continue;
            Rect r = boundingRect(cnt);
            Point2f c(r.x + r.width / 2.0f, r.y + r.height / 2.0f);
            detections.push_back(make_pair(c, r));
        }

        t1 = chrono::high_resolution_clock::now();
        tracker.update(detections);
        t2 = chrono::high_resolution_clock::now();
        total_tracker += chrono::duration<double, milli>(t2 - t1).count();
        
        t1 = chrono::high_resolution_clock::now();
        Mat vis = frame_up.clone();
        for (auto& t : tracker.get_tracks()) {
            Rect r = t.bbox;
            Point2f c = t.centroid;

            // draw track history
            vector<Point2f> pts(t.history.begin(), t.history.end());
            for (size_t i = 1; i < pts.size(); ++i) {
                line(vis, pts[i - 1], pts[i], COLOR_TRACK, 1);
            }

            Scalar color = t.confirmed ? COLOR_CONFIRMED : COLOR_PENDING;
            Rect r_clamped = r & Rect(0, 0, w_s, h_s);
            rectangle(vis, r_clamped, color, 2);
            putText(vis, "ID:" + to_string(t.id) + " H:" + to_string(t.hits) + " M:" + to_string(t.misses),
                Point(r_clamped.x, max(0, r_clamped.y - 6)), FONT_HERSHEY_SIMPLEX, 0.4, color, 1);

            csv_file << frame_idx << "," << t.id << "," << (int)c.x << "," << (int)c.y << ","
                << r.width << "," << r.height << "," << (int)t.confirmed << "\n";
        } 
        t2 = chrono::high_resolution_clock::now();
        total_draw += chrono::duration<double, milli>(t2 - t1).count();

        t1 = chrono::high_resolution_clock::now();
        writer.write(vis);
        t2 = chrono::high_resolution_clock::now();
        total_write += chrono::duration<double, milli>(t2 - t1).count();

        if (frame_idx % 10 == 0) {
            cout << "Frame " << frame_idx;
        }
    }

    csv_file.close();
    cap.release();
    writer.release();
    cout << "Done. Output: " << OUT_PATH << " CSV: " << OUT_CSV << endl;

    auto t_global_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_global_end - t_global_start).count();

    cout << "\n=====================================\n";
    cout << "Total processing time: " << total_sec << " sec\n";
    cout << "Avg time per frame: " << (total_sec / frame_idx) << " sec/frame\n";
    cout << "=====================================\n";

    cout << "\n=== Detailed timing (ms total) ===\n";
    cout << "Gray:        " << total_gray << endl;
    cout << "MOG2:        " << total_mog << endl;
    cout << "Threshold:   " << total_thresh << endl;
    cout << "Morph OPEN:  " << total_open << endl;
    cout << "Morph CLOSE: " << total_close << endl;
    cout << "Contours:    " << total_contours << endl;
    cout << "Tracker:     " << total_tracker << endl;
    cout << "Drawing:     " << total_draw << endl;
    cout << "Writer:      " << total_write << endl;

    cout << "\n=== Avg per frame (ms) ===\n";
    cout << "Gray:        " << total_gray / frame_idx << endl;
    cout << "MOG2:        " << total_mog / frame_idx << endl;
    cout << "Threshold:   " << total_thresh / frame_idx << endl;
    cout << "Morph OPEN:  " << total_open / frame_idx << endl;
    cout << "Morph CLOSE: " << total_close / frame_idx << endl;
    cout << "Contours:    " << total_contours / frame_idx << endl;
    cout << "Tracker:     " << total_tracker / frame_idx << endl;
    cout << "Drawing:     " << total_draw / frame_idx << endl;
    cout << "Writer:      " << total_write / frame_idx << endl;

    return 0;
}
