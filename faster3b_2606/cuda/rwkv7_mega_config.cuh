#pragma once

#include <cstdint>

namespace rwkv7_mega {

constexpr int kInstructionWords = 32;

enum Opcode : int32_t {
    OP_NOOP = 0,
    OP_EMB_LN0 = 1,
    OP_LN_MIX6_RKV = 2,
    OP_LOWRANK_GATES = 3,
    OP_KK_A_GATE = 4,
    OP_WKV = 5,
    OP_LNX_RKVRES_XG = 6,
    OP_ATT_OUT = 7,
    OP_CMIX_BLOCK = 8,
    OP_LN_OUT_HEAD = 9,
    OP_COPY_VEC = 10,
    OP_ROW1_LINEAR = 11,
    OP_ROW1_LINEAR_GROUP = 12,
    OP_LN_MIX6 = 13,
    OP_RKV_LINEAR_GROUP = 14,
    OP_KK_WKV_ROW_T1 = 15,
    OP_KK_WKV_LNX_T1 = 16,
    OP_ROW1_LINEAR_EXACT4 = 17,
    OP_COPY_VEC_REPEAT = 18,
    OP_LOWRANK_PRE = 19,
    OP_LOWRANK_V_FINISH = 20,
    OP_DIE = 255,
};

enum InstField : int32_t {
    F_ID = 0,
    F_OPCODE = 1,
    F_LAYER = 2,
    F_BATCH_START = 3,
    F_BATCH_COUNT = 4,
    F_TIME_START = 5,
    F_TIME_COUNT = 6,
    F_ROW_START = 7,
    F_ROW_COUNT = 8,
    F_POOL = 9,
    F_FLAGS = 10,
    F_COST_KIB = 11,
    F_COST_KOPS = 12,
    F_DEP_START = 13,
    F_DEP_COUNT = 14,
    F_INPUT_PAGE0 = 15,
    F_INPUT_PAGE1 = 16,
    F_INPUT_PAGE2 = 17,
    F_INPUT_PAGE3 = 18,
    F_OUTPUT_PAGE0 = 19,
    F_OUTPUT_PAGE1 = 20,
    F_WEIGHT_LAYOUT_ID = 21,
    F_LAUNCH_BLOCKS = 22,
    F_LAUNCH_THREADS = 23,
    F_WORK_COUNT = 24,
};

} // namespace rwkv7_mega
