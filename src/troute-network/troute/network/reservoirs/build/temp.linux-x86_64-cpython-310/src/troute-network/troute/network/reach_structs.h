#ifndef REACH_STRUCTS_H
#define REACH_STRUCTS_H
/*
    C Structures
*/
#include "musking/mc_reach_structs.h"
#include "reservoirs/levelpool/levelpool_structs.h"
#include "reservoirs/hybrid/hybrid_structs.h"
#include "reservoirs/rfc/rfc_structs.h"

typedef union {
  _MC_Reach mc_reach;
  _MC_Levelpool lp;
  _MC_Hybrid hybrid;
  _MC_RFC rfc;
} _ReachUnion;

typedef struct {
  _ReachUnion reach;
  int _num_segments; /* FIXME NOT USED HERE??? */
  long* _upstream_ids;
  int _num_upstream_ids;
  int type;
  long id;
} _Reach;

#endif //REACH_STRUCTS_H
