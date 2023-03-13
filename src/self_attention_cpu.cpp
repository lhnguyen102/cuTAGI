///////////////////////////////////////////////////////////////////////////////
// File:         self_attention_cpu.cpp
// Description:  CPU version for self attention
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 13, 2023
// Updated:      March 13, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2023 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "../include/self_attention_cpu.h"

void self_attention_forward_cpu(MultiHeadAttention &multi_head_att_state)
/*Multi-head self-attentiopn mecanism.

Args:
    mu_k: Mean of keys       (batch_size, num_heads, time_step, head_size)
    var_k: Variance of keys  (batch_size, num_heads, time_step, head_size)
    mu_q: Mean of query      (batch_size, num_heads, time_step, head_size)
    var_q: Variance of query (batch_size, num_heads, time_step, head_size)
    mu_v: Mean of value      (batch_size, num_heads, time_step, head_size)
    var_v: Variance of value (batch_size, num_heads, time_step, head_size)
    num_heads: Number of attention heads
    time_step: Time step
    head_size: Size of attention heads
    mu_att: Mean of attention      (batch_size, num_heads, time_step, head_size)
    var_att: Variance of attention (batch_size, num_heads, time_step, head_size)

*/
{
    // TO BE COMPLETED
}