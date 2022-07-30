///////////////////////////////////////////////////////////////////////////////
// File:         feature_availability.cpp
// Description:  Check the feature availability
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      June 05, 2022
// Updated:      July 30 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "../include/feature_availability.h"

void conv_cpu_support(Network &net)
/*CPU version does not support conv

Args:
    layers: All layer types of the network
    layer_names: Code name of each layer
*/
{
    if (is_conv(net.layers, net.layer_names)) {
        throw std::invalid_argument(
            "CPU version does not support conv layer - support_feature.cpp");
    }
}

void tconv_cpu_support(Network &net)
/*CPU version does not support conv
 */
{
    if (is_tconv(net.layers, net.layer_names)) {
        throw std::invalid_argument(
            "CPU version does not support transpose conv layer - "
            "support_feature.cpp");
    }
}

void derivative_support(Network &net)
/*CPU version does not support conv
 */
{
    if ((is_conv(net.layers, net.layer_names) ||
         is_tconv(net.layers, net.layer_names) ||
         is_leakyrelu(net.activations)) &&
        net.collect_derivative) {
        throw std::invalid_argument(
            "cuTAGI does not support the derivative calculations for conv. "
            "layer - "
            "support_feature.cpp");
    }
}

void full_cov_support(Network &net)
/*CPU version does not support conv
 */
{
    if ((is_conv(net.layers, net.layer_names) ||
         is_tconv(net.layers, net.layer_names)) &&
        net.is_full_cov) {
        throw std::invalid_argument(
            "cuTAGI does not support the full covariance for conv. "
            "layer - "
            "support_feature.cpp");
    }
}

void check_feature_availability(Network &net)
/* All feature messages that current CPU version does not support*/
{
    if (net.device.compare("cpu") == 0) {
        conv_cpu_support(net);
        tconv_cpu_support(net);
    }
    derivative_support(net);
    full_cov_support(net);
}