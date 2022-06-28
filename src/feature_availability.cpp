///////////////////////////////////////////////////////////////////////////////
// File:         feature_availability.cpp
// Description:  Check the feature availability
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      June 05, 2022
// Updated:      June 05 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "../include/feature_availability.h"

void conv_cpu_support(std::vector<int> &layer, LayerLabel &layer_names)
/*CPU version does not support conv

Args:
    layers: All layer types of the network
    layer_names: Code name of each layer
*/
{
    if (is_conv(layer, layer_names)) {
        throw std::invalid_argument(
            "CPU version does not support conv layer - support_feature.cpp");
    }
}

void tconv_cpu_support(std::vector<int> &layer, LayerLabel &layer_names)
/*CPU version does not support conv
 */
{
    if (is_tconv(layer, layer_names)) {
        throw std::invalid_argument(
            "CPU version does not support transpose conv layer - "
            "support_feature.cpp");
    }
}

void cpu_feature_availability(Network &net, LayerLabel &layer_names)
/* All feature messages that current CPU version does not support*/
{
    conv_cpu_support(net.layers, layer_names);
    tconv_cpu_support(net.layers, layer_names);
}