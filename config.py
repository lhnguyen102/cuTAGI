import numpy as np
import numpy.typing as npt
import copy

class ResnetBlock:
    layers: list=None
    filters: list=None
    kernels: list=None
    pad: list=None
    pad_types: list=None
    strides: list=None
    activations: list=None
    shorcuts: list=None
    t_kernel: int=None
    t_pad: int=None
    t_pad_type: int=None
    t_stride: int=None

class Config:
    def __init__(self,\
                 layers: list=None,\
                 nodes: list=None,\
                 kernels: list=None,\
                 strides: list=None,\
                 widths: list=None,\
                 heights: list=None,\
                 filters: list=None,
                 pads: list=None,\
                 pad_types: list=None,\
                 activations: list=None,\
                 shortcuts: list=None,
                 net_name: str=None) -> None:

        self.layers = layers
        self.nodes = nodes
        self.kernels = kernels
        self.strides = strides
        self.widths = widths
        self.heights = heights
        self.filters = filters
        self.pads = pads
        self.pad_types = pad_types
        self.activations = activations
        self.shortcuts = shortcuts
        self.net_name = net_name
        self.batch_size = 2
        self.sigma_v = 4

    def write_txt(self):

        # Create data
        net = {
                "layers": self.layers,\
                "nodes": self.nodes,\
                "kernels": self.kernels,\
                "strides": self.strides,\
                "widths": self.widths,\
                "heights": self.heights,\
                "filters": self.filters,\
                "pads": self.pads,\
                "pad_types": self.pad_types,\
                "activations": self.activations,\
                "shortcuts": self.shortcuts,\
                "batch_size": self.batch_size,\
                "sigma_v": self.sigma_v
                }

        # Writing to .json
        file_name = f"{self.net_name}.txt" 
        with open(file_name, "w") as f:
            for key, value in net.items():
                f.write('%s:%s\n' %(key, value))

    def resnet18_10_classes(self, img_size: npt.NDArray, ny: int) -> None:

        # Normalizaiton
        norm_layer = 6
        factor = 2
        num_rep = 2
        resnet = []

        # Input
        rs_block_0 = ResnetBlock()
        rs_block_0.layers      = [2,                  2,      norm_layer]
        rs_block_0.filters     = [img_size[2],        32,      32]
        rs_block_0.kernels     = [3,                  1,      3]
        rs_block_0.pads        = [1,                  0,      1]
        rs_block_0.pad_types   = [1,                  0,      1]
        rs_block_0.strides     = [1,                  0,      1]
        rs_block_0.nodes       = [np.prod(img_size),  0,      0]
        rs_block_0.activations = [0,                  4,      0]
        rs_block_0.shortcuts   = [-1,                -1,      1]
        resnet.append(rs_block_0)

        # Block 1
        rs_block_1 = ResnetBlock()
        rs_block_1.layers      = [2,    norm_layer,     2,     norm_layer]
        rs_block_1.filters     = [32,    32,            32,     32]
        rs_block_1.kernels     = [1,    3,              1,      3]
        rs_block_1.pads        = [0,    1,              0,      1]
        rs_block_1.pad_types   = [0,    1,              0,      1]
        rs_block_1.strides     = [0,    1,              0,      1]
        rs_block_1.nodes       = [0,    0,              0,      0]
        rs_block_1.activations = [4,    0,              4,      0]
        rs_block_1.shortcuts   = [-1,  -1,             -1,     -1]
        rs_block_1.t_kernel    = 3
        rs_block_1.t_pad       = 1
        rs_block_1.t_pad_type  = 2
        rs_block_1.t_stride    = 2

        rs_block_ref = copy.copy(rs_block_1)

        rs_block_1 = self.copy_resnet_block(rs_block_1, num_rep)
        resnet.append(rs_block_1)

        # Block 2
        rs_block_2 = copy.copy(rs_block_ref)
        rs_block_2.filters = [l * factor for l in rs_block_ref.filters]
        rs_block_2 = self.copy_resnet_block(rs_block_2, num_rep)
        resnet.append(rs_block_2)

        # Block 3
        rs_block_3 = copy.copy(rs_block_ref)
        rs_block_3.filters = [l * 2 * factor for l in rs_block_ref.filters]
        rs_block_3 = self.copy_resnet_block(rs_block_3, num_rep)
        resnet.append(rs_block_3)

        # Block 4
        rs_block_4 = copy.copy(rs_block_ref)
        rs_block_4.filters = [l * 4 * factor for l in rs_block_ref.filters]
        rs_block_4.t_kernel    = 4
        rs_block_4.t_pad       = 0
        rs_block_4.t_pad_type  = 2
        rs_block_4.t_stride    = 1
        rs_block_4 = self.copy_resnet_block(rs_block_4, num_rep)
        resnet.append(rs_block_4)


        # Output
        rs_block_5 = ResnetBlock()
        rs_block_5.layers      = [4,                                 1]
        rs_block_5.filters     = [rs_block_4.filters[1] * factor,    1]
        rs_block_5.kernels     = [1,                                 1]
        rs_block_5.pads        = [0,                                 0]
        rs_block_5.pad_types   = [0,                                 0]
        rs_block_5.strides     = [0,                                 0]
        rs_block_5.nodes       = [0,                                 ny]
        rs_block_5.activations = [0,                                 0]
        rs_block_5.shortcuts   = [-1,                               -1]
        resnet.append(rs_block_5)

        # Convert to list
        self.layers = np.concatenate([l.layers for l in resnet]).tolist()
        self.filters = np.concatenate([f.filters for f in resnet]).tolist()
        self.kernels = np.concatenate([k.kernels for k in resnet]).tolist()
        self.pads = np.concatenate([p.pads for p in resnet]).tolist()
        self.pad_types = np.concatenate([pt.pad_types for pt in resnet]).tolist()
        self.strides = np.concatenate([s.strides for s in resnet]).tolist()
        self.nodes = np.concatenate([n.nodes for n in resnet]).tolist()
        self.activations = np.concatenate([a.activations for a in resnet]).tolist()
        self.shortcuts = np.concatenate([st.shortcuts for st in resnet]).tolist()
        self.widths = np.zeros((len(self.layers), ), dtype=int).tolist()
        self.heights = np.zeros((len(self.layers),), dtype=int).tolist()
        self.widths[0] = img_size[0]
        self.heights[0] = img_size[1]

        # Shortcut
        shortcut_layers = np.where(np.array(self.shortcuts) == 1)[0]
        shortcut_layers[1:] = shortcut_layers[0:-1]
        shortcut_layers[0] = -1
        shortcut_array = np.array(self.shortcuts)
        shortcut_array[np.array(self.shortcuts) == 1] = shortcut_layers
        self.shortcuts = list(shortcut_array)


    def copy_resnet_block(self, rs_block: ResnetBlock,\
            num_rep: int) -> ResnetBlock:
        """
        Repeat the resnet block multiple times

        """
        n_layer = len(rs_block.layers)
        rs_block.layers = np.tile(rs_block.layers, num_rep)
        rs_block.filters = np.tile(rs_block.filters, num_rep)
        rs_block.kernels = np.tile(rs_block.kernels, num_rep)
        rs_block.kernels[-1] = rs_block.t_kernel
        rs_block.pads = np.tile(rs_block.pads, num_rep)
        rs_block.pads[-1] = rs_block.t_pad
        rs_block.pad_types = np.tile(rs_block.pad_types, num_rep)
        rs_block.pad_types[-1] = rs_block.t_pad_type
        rs_block.strides = np.tile(rs_block.strides, num_rep)
        rs_block.strides[-1] = rs_block.t_stride
        rs_block.nodes = np.tile(rs_block.nodes, num_rep)
        rs_block.activations = np.tile(rs_block.activations, num_rep)
        rs_block.shortcuts = np.tile(rs_block.shortcuts, num_rep)
        rs_block.shortcuts[(n_layer-1)::n_layer] = 1

        return rs_block

def main():
#    layers = [1, 2, 2, 1]
#    nodes = [100, 0, 0, 0]
#    kernels = [1, 1, 1, 1]
#    strides = [0, 0, 0, 0]
#    widths = [0, 0, 0, 0]
#    heights = [0, 0, 0, 0]
#    filters = [1, 1, 1, 1]
#    pads = [0, 0, 0, 0]
#    pad_types =[1, 1, 1, 1]
#    shortcuts = [-1, -1, -1, -1]
#    net_name = "test"
#    
#    cfg = Config(layers=layers,\
#                 nodes=nodes,\
#                 kernels=kernels,\
#                 strides=strides,\
#                 widths=widths,\
#                 heights=heights,\
#                 filters=filters,\
#                 pads=pads,\
#                 pad_types=pad_types,\
#                 shortcuts=shortcuts,\
#                 net_name=net_name)

    net_name = 'resnet18_10_classes'
    ny = 11
    img_size = [32, 32, 3]
    cfg = Config(net_name=net_name)
    cfg.resnet18_10_classes(img_size, ny)

    cfg.write_txt()

if __name__ == "__main__":
    main()

