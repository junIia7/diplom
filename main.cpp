#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <iostream>
#include <fstream>
#include <deque>
#include <map>
#include <cmath>
#include <chrono>

using namespace cv;
using namespace std;

// ---------------- Parameters ----------------
const string VIDEO_PATH = "./video.mp4";
const string OUT_PATH = "./output_cuda.mp4";
const string OUT_CSV = "./detections_cuda.csv";

const int MOG_HISTORY = 30;
const double MOG_VAR_THRESHOLD = 25;
const bool MOG_DETECT_SHADOWS = false;

const double MIN_AREA = 200;
const double MAX_AREA = 2000;
const int N_CONFIRM = 10;
const int MAX_MISSES = 6;
const double DIST_THRESH = 60.0;

const double UPSCALE = 4.0;

// Morphology kernels (CPU создаём, CUDA использует их внутри фильтра)
Mat KERNEL_OPEN = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
Mat KERNEL_CLOSE = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));

// Colors
Scalar COLOR_PENDING(0, 255, 255);
Scalar COLOR_CONFIRMED(0, 0, 255);
Scalar COLOR_TRACK(255, 0, 0);

// ---------------- Tracker ----------------
struct Track {
    int id;
    Point2f centroid;
    Rect bbox;
    int hits;
    int misses;
    deque<Point2f> history;
    bool confirmed;

    Track() {}
    Track(int tid, Point2f c, Rect b)
        : id(tid), centroid(c), bbox(b),
        hits(1), misses(0), confirmed(false)
    {
        history.push_back(c);
    }
};

class SimpleTracker {
private:
    int next_id = 1;
    map<int, Track> tracks;

    double dist(Point2f a, Point2f b) {
        return hypot(a.x - b.x, a.y - b.y);
    }

public:
    void update(vector<pair<Point2f, Rect>>& detections) {

        set<int> assigned;
        vector<int> unmatched;

        for (auto& kv : tracks)
            unmatched.push_back(kv.first);

        for (size_t i = 0; i < detections.size(); ++i) {

            Point2f c = detections[i].first;
            Rect r = detections[i].second;

            int best_id = -1;
            double best_dist = 1e9;

            for (int tid : unmatched) {
                double d = dist(c, tracks[tid].centroid);
                if (d < best_dist) {
                    best_dist = d;
                    best_id = tid;
                }
            }

            if (best_id != -1 && best_dist < DIST_THRESH) {

                auto& t = tracks[best_id];
                t.centroid = c;
                t.bbox = r;
                t.history.push_back(c);
                if (t.history.size() > 50) t.history.pop_front();

                t.hits++;
                t.misses = 0;
                if (t.hits >= N_CONFIRM)
                    t.confirmed = true;

                unmatched.erase(remove(unmatched.begin(), unmatched.end(), best_id), unmatched.end());
                assigned.insert(i);
            }
        }

        for (size_t i = 0; i < detections.size(); ++i) {
            if (!assigned.count(i)) {
                tracks[next_id] = Track(next_id, detections[i].first, detections[i].second);
                next_id++;
            }
        }

        for (int tid : unmatched) {
            tracks[tid].misses++;
            if (tracks[tid].misses > MAX_MISSES)
                tracks.erase(tid);
        }
    }

    vector<Track> get_tracks() {
        vector<Track> out;
        for (auto& kv : tracks)
            out.push_back(kv.second);
        return out;
    }
};

// ---------------- MAIN ----------------
int main() {
    string qqq;
    if (cuda::getCudaEnabledDeviceCount() == 0) {
        cerr << "CUDA device not found!\n";
        return -1;
    }

    cuda::setDevice(0);

    cout << "CUDA device count: "
        << cuda::getCudaEnabledDeviceCount() << endl;

    VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        cerr << "Cannot open video\n";
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    if (fps == 0) fps = 25.0;

    int w = cap.get(CAP_PROP_FRAME_WIDTH);
    int h = cap.get(CAP_PROP_FRAME_HEIGHT);

    //VideoWriter writer(OUT_PATH,
    //    VideoWriter::fourcc('m', 'p', '4', 'v'),
    //    fps, Size(w*UPSCALE, h*UPSCALE));

    ofstream csv_file(OUT_CSV);
    csv_file << "frame,id,x,y,w,h,confirmed\n";

    // CUDA objects
    Ptr<cuda::BackgroundSubtractorMOG2> mog =
        cuda::createBackgroundSubtractorMOG2(
            MOG_HISTORY,
            MOG_VAR_THRESHOLD,
            MOG_DETECT_SHADOWS);

    auto morphOpen = cuda::createMorphologyFilter(
        MORPH_OPEN, CV_8UC1, KERNEL_OPEN);

    auto morphClose = cuda::createMorphologyFilter(
        MORPH_CLOSE, CV_8UC1, KERNEL_CLOSE);

    SimpleTracker tracker;

    // Profiling
    double total_upload = 0, total_gray = 0, total_mog = 0;
    double total_thresh = 0, total_open = 0, total_close = 0;
    double total_download = 0, total_contours = 0;
    double total_tracker = 0, total_draw = 0, total_write = 0;

    auto global_start = chrono::high_resolution_clock::now();

    Mat frame;
    int frame_idx = 0;

    while (cap.read(frame)) {
        if (UPSCALE != 1.0) {
            cv::resize(frame, frame, cv::Size(), UPSCALE, UPSCALE, cv::INTER_LINEAR);
        }

        frame_idx++;

        cuda::GpuMat d_frame, d_gray, d_fgmask;

        auto t1 = chrono::high_resolution_clock::now();
        d_frame.upload(frame);
        auto t2 = chrono::high_resolution_clock::now();
        total_upload += chrono::duration<double, milli>(t2 - t1).count();

        t1 = chrono::high_resolution_clock::now();
        cuda::cvtColor(d_frame, d_gray, COLOR_BGR2GRAY);
        t2 = chrono::high_resolution_clock::now();
        total_gray += chrono::duration<double, milli>(t2 - t1).count();

        t1 = chrono::high_resolution_clock::now();
        mog->apply(d_gray, d_fgmask);
        t2 = chrono::high_resolution_clock::now();
        total_mog += chrono::duration<double, milli>(t2 - t1).count();

        t1 = chrono::high_resolution_clock::now();
        cuda::threshold(d_fgmask, d_fgmask, 127, 255, THRESH_BINARY);
        t2 = chrono::high_resolution_clock::now();
        total_thresh += chrono::duration<double, milli>(t2 - t1).count();

        t1 = chrono::high_resolution_clock::now();
        morphOpen->apply(d_fgmask, d_fgmask);
        t2 = chrono::high_resolution_clock::now();
        total_open += chrono::duration<double, milli>(t2 - t1).count();

        t1 = chrono::high_resolution_clock::now();
        morphClose->apply(d_fgmask, d_fgmask);
        t2 = chrono::high_resolution_clock::now();
        total_close += chrono::duration<double, milli>(t2 - t1).count();

        Mat fgmask;
        t1 = chrono::high_resolution_clock::now();
        d_fgmask.download(fgmask);
        t2 = chrono::high_resolution_clock::now();
        total_download += chrono::duration<double, milli>(t2 - t1).count();

        vector<vector<Point>> contours;
        t1 = chrono::high_resolution_clock::now();
        findContours(fgmask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        t2 = chrono::high_resolution_clock::now();
        total_contours += chrono::duration<double, milli>(t2 - t1).count();

        vector<pair<Point2f, Rect>> detections;
        for (auto& c : contours) {
            double area = contourArea(c);
            if (area < MIN_AREA || area > MAX_AREA) continue;
            Rect r = boundingRect(c);
            Point2f center(r.x + r.width / 2.0f,
                r.y + r.height / 2.0f);
            detections.push_back({ center,r });
        }

        t1 = chrono::high_resolution_clock::now();
        tracker.update(detections);
        t2 = chrono::high_resolution_clock::now();
        total_tracker += chrono::duration<double, milli>(t2 - t1).count();

        /*t1 = chrono::high_resolution_clock::now();
        Mat vis = frame.clone();

        for (auto& t : tracker.get_tracks()) {

            for (size_t i = 1; i < t.history.size(); ++i)
                line(vis, t.history[i - 1], t.history[i], COLOR_TRACK, 1);

            Scalar col = t.confirmed ? COLOR_CONFIRMED : COLOR_PENDING;
            rectangle(vis, t.bbox, col, 2);

            putText(vis, "ID:" + to_string(t.id),
                Point(t.bbox.x, t.bbox.y - 5),
                FONT_HERSHEY_SIMPLEX, 0.4, col, 1);

            csv_file << frame_idx << ","
                << t.id << ","
                << t.centroid.x << ","
                << t.centroid.y << ","
                << t.bbox.width << ","
                << t.bbox.height << ","
                << t.confirmed << "\n";
        }

        t2 = chrono::high_resolution_clock::now();
        total_draw += chrono::duration<double, milli>(t2 - t1).count();*/

        //t1 = chrono::high_resolution_clock::now();
        //writer.write(vis);
        //t2 = chrono::high_resolution_clock::now();
        //total_write += chrono::duration<double, milli>(t2 - t1).count();

        if (frame_idx % 10 == 0)
            cout << "Frame " << frame_idx << endl;
    }

    auto global_end = chrono::high_resolution_clock::now();
    double total_sec =
        chrono::duration<double>(global_end - global_start).count();

    cout << "\n================ CUDA PROFILING ================\n";
    cout << "Total time: " << total_sec << " sec\n";
    cout << "Avg time per frame: "
        << total_sec / frame_idx << " sec/frame\n";

    cout << "\n--- Avg per frame (ms) ---\n";
    cout << "Upload:     " << total_upload / frame_idx << endl;
    cout << "Gray:       " << total_gray / frame_idx << endl;
    cout << "MOG2:       " << total_mog / frame_idx << endl;
    cout << "Threshold:  " << total_thresh / frame_idx << endl;
    cout << "MorphOpen:  " << total_open / frame_idx << endl;
    cout << "MorphClose: " << total_close / frame_idx << endl;
    cout << "Download:   " << total_download / frame_idx << endl;
    cout << "Contours:   " << total_contours / frame_idx << endl;
    cout << "Tracker:    " << total_tracker / frame_idx << endl;
    cout << "Draw:       " << total_draw / frame_idx << endl;
    cout << "Writer:     " << total_write / frame_idx << endl;

    cap.release();
    //writer.release();
    csv_file.close();

    cin >> qqq;

    return 0;
}
