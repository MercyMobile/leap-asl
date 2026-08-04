/* Grab one stereo IR image set from the Leap and write it as two PGM files.
 * Skips the first N image events so auto-exposure can settle.
 * Usage: leap-grab [output-prefix] [skip-frames]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "LeapC.h"
#include "ExampleConnection.h"

static const char* g_prefix = "leap";
static int g_skip = 40;
static int g_seen = 0;
static volatile int g_done = 0;

static void OnConnect(void){ printf("Connected.\n"); }

static void OnDevice(const LEAP_DEVICE_INFO *props){
  printf("Found device %s.\n", props->serial);
}

static void write_pgm(const char* path, const LEAP_IMAGE* img){
  unsigned w = img->properties.width, h = img->properties.height;
  const unsigned char* px = (const unsigned char*)img->data + img->offset;
  FILE* f = fopen(path, "wb");
  if(!f){ perror("fopen"); return; }
  fprintf(f, "P5\n%u %u\n255\n", w, h);
  fwrite(px, 1, (size_t)w*h, f);
  fclose(f);

  /* stats so the image can be described without looking at it */
  unsigned long sum = 0; unsigned mn = 255, mx = 0;
  unsigned long hist[4] = {0,0,0,0};
  for(size_t i = 0; i < (size_t)w*h; i++){
    unsigned char v = px[i];
    sum += v; if(v < mn) mn = v; if(v > mx) mx = v;
    hist[v >> 6]++;
  }
  double mean = (double)sum / ((double)w*h);
  printf("  %s: %ux%u bpp=%u  min=%u max=%u mean=%.1f\n",
         path, w, h, img->properties.bpp, mn, mx, mean);
  printf("     tone spread: dark(0-63)=%.1f%%  low(64-127)=%.1f%%  "
         "high(128-191)=%.1f%%  bright(192-255)=%.1f%%\n",
         100.0*hist[0]/((double)w*h), 100.0*hist[1]/((double)w*h),
         100.0*hist[2]/((double)w*h), 100.0*hist[3]/((double)w*h));
}

static void OnImage(const LEAP_IMAGE_EVENT *ev){
  if(g_done) return;
  if(g_seen++ < g_skip) return;
  char path[512];
  printf("Captured image set, frame %lli:\n", (long long)ev->info.frame_id);
  snprintf(path, sizeof path, "%s_left.pgm",  g_prefix);
  write_pgm(path, &ev->image[0]);
  snprintf(path, sizeof path, "%s_right.pgm", g_prefix);
  write_pgm(path, &ev->image[1]);
  g_done = 1;
}

int main(int argc, char** argv){
  if(argc > 1) g_prefix = argv[1];
  if(argc > 2) g_skip = atoi(argv[2]);

  ConnectionCallbacks.on_connection   = &OnConnect;
  ConnectionCallbacks.on_device_found = &OnDevice;
  ConnectionCallbacks.on_image        = &OnImage;

  LEAP_CONNECTION *connection = OpenConnection();
  LeapSetPolicyFlags(*connection, eLeapPolicyFlag_Images, 0);

  for(int i = 0; i < 300 && !g_done; i++) usleep(50000);  /* up to 15s */

  CloseConnection();
  DestroyConnection();
  if(!g_done){ fprintf(stderr, "No image received.\n"); return 1; }
  printf("Done.\n");
  return 0;
}
