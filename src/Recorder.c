/* Head-to-head recorder: for each sampled instant, save the Leap IR image AND
 * the Leap's own native hand-tracking verdict for the same frame_id.
 *
 * Usage: leap-rec <outdir> <num_samples> <interval_ms>
 * Writes <outdir>/f<frame_id>.pgm  and  <outdir>/native.csv
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#include "LeapC.h"
#include "ExampleConnection.h"

#define RING 512

static char g_dir[256] = ".";
static int  g_want = 30, g_interval_ms = 1000, g_saved = 0;
static long long g_last_save_us = 0;
static volatile int g_done = 0;

/* ring of recent tracking verdicts, keyed by frame id */
static struct { long long fid; int nhands; float conf; float px, py, pz; int grab; } g_ring[RING];
static int g_ring_n = 0;

static FILE* g_csv = NULL;

static long long now_us(void){
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec*1000000 + ts.tv_nsec/1000;
}

static void OnConnect(void){ printf("Connected.\n"); }
static void OnDevice(const LEAP_DEVICE_INFO *p){ printf("Device %s.\n", p->serial); }

static void OnFrame(const LEAP_TRACKING_EVENT *f){
  int i = g_ring_n++ % RING;
  g_ring[i].fid = (long long)f->info.frame_id;
  g_ring[i].nhands = f->nHands;
  if(f->nHands > 0){
    g_ring[i].conf = f->pHands[0].confidence;
    g_ring[i].px = f->pHands[0].palm.position.x;
    g_ring[i].py = f->pHands[0].palm.position.y;
    g_ring[i].pz = f->pHands[0].palm.position.z;
    g_ring[i].grab = (int)(f->pHands[0].grab_strength * 100);
  } else {
    g_ring[i].conf = 0; g_ring[i].px = g_ring[i].py = g_ring[i].pz = 0; g_ring[i].grab = 0;
  }
}

/* nearest tracking verdict to this image's frame id */
static int lookup(long long fid, int *nh, float *conf, float *px, float *py, float *pz, int *grab){
  long long best = -1; int bi = -1;
  int n = g_ring_n < RING ? g_ring_n : RING;
  for(int i = 0; i < n; i++){
    long long d = g_ring[i].fid - fid; if(d < 0) d = -d;
    if(best < 0 || d < best){ best = d; bi = i; }
  }
  if(bi < 0 || best > 5) return 0;          /* require within 5 frames */
  *nh = g_ring[bi].nhands; *conf = g_ring[bi].conf;
  *px = g_ring[bi].px; *py = g_ring[bi].py; *pz = g_ring[bi].pz; *grab = g_ring[bi].grab;
  return 1;
}

static void OnImage(const LEAP_IMAGE_EVENT *ev){
  if(g_done) return;
  long long t = now_us();
  if(g_last_save_us && (t - g_last_save_us) < (long long)g_interval_ms*1000) return;
  g_last_save_us = t;

  long long fid = (long long)ev->info.frame_id;
  char path[512];
  for(int eye = 0; eye < 2; eye++){
    const LEAP_IMAGE* img = &ev->image[eye];
    unsigned w = img->properties.width, h = img->properties.height;
    const unsigned char* px = (const unsigned char*)img->data + img->offset;
    snprintf(path, sizeof path, "%s/f%lld_%c.pgm", g_dir, fid, eye ? 'R' : 'L');
    FILE* f = fopen(path, "wb");
    if(f){
      fprintf(f, "P5\n%u %u\n255\n", w, h);
      fwrite(px, 1, (size_t)w*h, f);
      fclose(f);
    }
    /* dump the 64x64 distortion grid once per eye -- this is the real calibration */
    static int dumped[2] = {0, 0};
    if(!dumped[eye] && img->distortion_matrix){
      snprintf(path, sizeof path, "%s/distortion_%c.txt", g_dir, eye ? 'R' : 'L');
      FILE* d = fopen(path, "w");
      if(d){
        fprintf(d, "# 64x64 grid, rows=v, cols=u, each entry 'x y' in normalized image coords\n");
        for(int r = 0; r < LEAP_DISTORTION_MATRIX_N; r++){
          for(int c = 0; c < LEAP_DISTORTION_MATRIX_N; c++)
            fprintf(d, "%.6f %.6f ", img->distortion_matrix->matrix[r][c].x,
                                     img->distortion_matrix->matrix[r][c].y);
          fprintf(d, "\n");
        }
        fclose(d);
        dumped[eye] = 1;
        printf("  wrote distortion grid for %s eye\n", eye ? "right" : "left");
      }
    }
  }

  int nh = 0, grab = 0; float conf = 0, ppx = 0, ppy = 0, ppz = 0;
  int ok = lookup(fid, &nh, &conf, &ppx, &ppy, &ppz, &grab);
  fprintf(g_csv, "%lld,%d,%d,%.3f,%.1f,%.1f,%.1f,%d\n",
          fid, ok, nh, conf, ppx, ppy, ppz, grab);
  fflush(g_csv);

  g_saved++;
  printf("  [%2d/%2d] frame %lld  leap_native=%s", g_saved, g_want, fid,
         nh > 0 ? "HAND" : "none");
  if(nh > 0) printf("  conf=%.2f palm=(%.0f,%.0f,%.0f)mm grab=%d%%", conf, ppx, ppy, ppz, grab);
  printf("\n");
  fflush(stdout);
  if(g_saved >= g_want) g_done = 1;
}

int main(int argc, char** argv){
  if(argc > 1) snprintf(g_dir, sizeof g_dir, "%s", argv[1]);
  if(argc > 2) g_want = atoi(argv[2]);
  if(argc > 3) g_interval_ms = atoi(argv[3]);
  /* Optional 4th arg: "hmd" or "screentop" selects the tracking optimisation.
     The service ships three exposure/tracking policies -- desktop, HMD and
     screentop -- and DESKTOP IS THE DEFAULT. Desktop assumes a hand hovering
     just above a puck sitting on a table, which is why range collapses past
     ~10in. HMD is what a headset mount uses: looking outward at hands held at
     arm's length and beyond. Cisco ran this controller on an HTC Vive and got
     far more than two feet, which is the observation that led here.

     MEASURED RESULT: it makes no difference. Four interleaved runs, 50 frames
     per mode, hand at ~15in -- leapd found hands in 0/50 under BOTH desktop and
     HMD. An earlier single run suggested 0/25 -> 12/25; that was noise. The
     switch is kept because it is a legitimate option, not because it works.
     Cisco's Vive range is explained by software, not policy: he ran Orion, and
     Orion V4 was never released for Linux. */
  uint64_t policy = eLeapPolicyFlag_Images;
  if(argc > 4){
    if(!strcmp(argv[4], "hmd"))            policy |= eLeapPolicyFlag_OptimizeHMD;
    else if(!strcmp(argv[4], "screentop")) policy |= eLeapPolicyFlag_OptimizeScreenTop;
  }
  mkdir(g_dir, 0775);

  char csvp[512]; snprintf(csvp, sizeof csvp, "%s/native.csv", g_dir);
  g_csv = fopen(csvp, "w");
  if(!g_csv){ perror("csv"); return 1; }
  fprintf(g_csv, "frame_id,matched,leap_nhands,leap_conf,palm_x,palm_y,palm_z,grab_pct\n");

  ConnectionCallbacks.on_connection   = &OnConnect;
  ConnectionCallbacks.on_device_found = &OnDevice;
  ConnectionCallbacks.on_frame        = &OnFrame;
  ConnectionCallbacks.on_image        = &OnImage;

  LEAP_CONNECTION *conn = OpenConnection();
  LeapSetPolicyFlags(*conn, policy, 0);
  if(policy != eLeapPolicyFlag_Images)
    printf("tracking optimisation: %s\n", argv[4]);

  int guard = 0;
  while(!g_done && guard++ < 4000) usleep(50000);   /* up to 200s */

  fclose(g_csv);
  CloseConnection();
  DestroyConnection();
  printf("Saved %d samples to %s\n", g_saved, g_dir);
  return 0;
}
