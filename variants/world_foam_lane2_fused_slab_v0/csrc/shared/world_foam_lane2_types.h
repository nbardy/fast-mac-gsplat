#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef uint WF2UInt32;
typedef metal::packed_float3 WF2Float3;
#else
#include <stdint.h>
typedef uint32_t WF2UInt32;
typedef struct WF2Float3 {
  float x;
  float y;
  float z;
} WF2Float3;
#endif

#define WF2_BUFFER_BEAMS 0
#define WF2_BUFFER_GRID 1
#define WF2_BUFFER_COUNTS 2
#define WF2_BUFFER_GLOBAL_EVENT_COUNT 3
#define WF2_BUFFER_EVENTS 4

#define WF2_POWER_BUFFER_BOUNDARIES 0
#define WF2_POWER_BUFFER_BEAMS 1
#define WF2_POWER_BUFFER_CONFIG 2
#define WF2_POWER_BUFFER_COUNTS 3

#define WF2_EVENT_AXIS_U 0u
#define WF2_EVENT_AXIS_V 1u
#define WF2_EVENT_AXIS_T 2u

#define WF2_GRID_FLAG_WRITE_EVENTS 1u

#define WF2_COUNT_FLAG_INVALID_BEAM 1u
#define WF2_COUNT_FLAG_EVENT_OVERFLOW 2u
#define WF2_COUNT_FLAG_INVALID_DENOMINATOR 4u

#define WF2_EVENT_FLAG_POSITIVE_DIRECTION 1u
#define WF2_NO_EVENT_INDEX 0xFFFFFFFFu

typedef struct WF2ScreenTimeBeam {
  WF2Float3 start_uvt;
  WF2Float3 end_uvt;
  float radius_px;
  float opacity_hint;
  WF2UInt32 payload_id;
  WF2UInt32 flags;
} WF2ScreenTimeBeam;

typedef struct WF2GridConfig {
  WF2UInt32 beam_count;
  WF2UInt32 tile_count_u;
  WF2UInt32 tile_count_v;
  WF2UInt32 tile_count_t;
  float tile_size_u;
  float tile_size_v;
  float tile_size_t;
  WF2UInt32 event_capacity;
  WF2UInt32 flags;
  WF2UInt32 reserved0;
  WF2UInt32 reserved1;
  WF2UInt32 reserved2;
} WF2GridConfig;

typedef struct WF2BeamEventCount {
  WF2UInt32 beam_id;
  WF2UInt32 payload_id;
  WF2UInt32 u_crossings;
  WF2UInt32 v_crossings;
  WF2UInt32 t_crossings;
  WF2UInt32 total_crossings;
  WF2UInt32 first_event_index;
  WF2UInt32 flags;
} WF2BeamEventCount;

typedef struct WF2BoundaryEvent {
  WF2UInt32 beam_id;
  WF2UInt32 payload_id;
  WF2UInt32 axis;
  WF2UInt32 boundary_index;
  float s;
  WF2Float3 uvt;
  WF2UInt32 flags;
  WF2UInt32 reserved0;
} WF2BoundaryEvent;

typedef struct WF2PowerBoundary3D {
  float nx;
  float nz;
  float nt;
  float b;
  WF2UInt32 left_site;
  WF2UInt32 right_site;
  WF2UInt32 reserved0;
  WF2UInt32 reserved1;
} WF2PowerBoundary3D;

typedef struct WF2PowerBeamSlab {
  float u_center;
  float t0;
  float t1;
  float near_depth;
  float far_depth;
  WF2UInt32 payload_id;
  WF2UInt32 flags;
  WF2UInt32 reserved0;
} WF2PowerBeamSlab;

typedef struct WF2PowerBoundaryConfig {
  WF2UInt32 boundary_count;
  WF2UInt32 beam_count;
  float camera_velocity_x;
  float invalid_epsilon;
} WF2PowerBoundaryConfig;

typedef struct WF2PowerBoundaryCount {
  WF2UInt32 beam_id;
  WF2UInt32 payload_id;
  WF2UInt32 boundary_event_count;
  WF2UInt32 invalid_denominator_count;
  WF2UInt32 flags;
  WF2UInt32 reserved0;
  WF2UInt32 reserved1;
  WF2UInt32 reserved2;
} WF2PowerBoundaryCount;
