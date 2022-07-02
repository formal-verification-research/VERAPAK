from . import vnnlib_base

import re
import numpy as np

class NonMaximalVNNLibError(ValueError):
    pass

class VNNLib():
    def __init__(self, filename):
        vnn_data = read_vnnlib_simple(filename)
        self.inputs = np.transpose(np.array(vnn_data[0][0]))
        self.mat = [] # Matrix
        self.rhs = [] # Right hand side
        for output in vnn_data[0][1]:
            self.mat.append(output[0])
            self.rhs.append(output[1])
    
    def is_maximal(self):
        if not hasattr(self, "_maximal_class"):
            self._maximal_class = -1 # Any breaks from here on will leave self._maximal_class as -1

            rhs = np.array(self.rhs).sum(axis=1) # Right-hand side should have all zeros -- otherwise, we are messing with constants, too
            if not rhs.all() == 0: # All RHS values should always be 0
                return False

            mat = np.array(self.mat).sum(axis=1) #/ 1 -1  0  0  .\
                                                 #| 1  0 -1  0  .| vstack(ones(n), -identity(n, n)).vshuffle()
                                                 #| 1  0  0 -1  .|
                                                 #| 1  0  0  0  .| (Vertical order will not be... orderly?)
                                                 #\ .  .  .  .  ./
            if len(mat.shape) != 2: # Valid MATs will always be 2D
                return False
            
            maximal_idx = -1
            compared = [False] * mat.shape[1]
            for i in range(mat.shape[0]):
                first = -1
                second = -1
                for j in range(mat.shape[1]):
                    el = mat[i][j]
                    if el == 1 and first == -1:
                        first = j
                    elif el == -1 and second == -1:
                        second = j
                    elif el != 0:
                        return False # Multi-variable comparison - not allowed in VERAPAK
                
                if first == -1 or second == -1:
                    return False # Single or no-variable comparison
                
                if maximal_idx == -1:
                    maximal_idx = first
                elif maximal_idx != first:
                    return False # No consistently maximal variable
                
                compared[second] = True
            for i in range(mat.shape[1]):
                if i == maximal_idx and compared[i]:
                    return False # Should not be comparing another variable with the maximal variable
                elif i != maximal_idx and not compared[i]:
                    print(str(maximal_idx) + "  " + str(i))
                    return False # Should compare all other variables

            self._maximal_class = maximal_idx # Only set it to something other than -1 *after* we have returned False in all bad cases

        return self._maximal_class != -1

    def get_centerpoint(self):
        if not hasattr(self, "_centerpoint"):
            self._centerpoint = (self.inputs[0] + self.inputs[1]) / 2

        return self._centerpoint

    def get_radii(self):
        if not hasattr(self, "_radii"):
            cp = self.get_centerpoint()
            self._radii = np.abs(np.subtract(self.inputs[1], cp))
        
        return self._radii

    def get_intended_class(self):
        if not self.is_maximal():
            raise NonMaximalVNNLibError("Cannot get the \"intended\" class of a non-maximal VNNLib")
        else:
            return self._maximal_class

    def get_domain(self):
        return self.inputs


def read_vnnlib_simple(filename):
    vars_in = 0
    vars_out = 0
    statements = vnnlib_base.read_statements(filename)
    regex_declare_in = re.compile(r"^\(declare-const X_(\S+) Real\)$")
    regex_declare_out = re.compile(r"^\(declare-const Y_(\S+) Real\)$")
    for statement in statements:
        if regex_declare_in.match(statement):
            vars_in += 1
        elif regex_declare_out.match(statement):
            vars_out += 1
    return vnnlib_base.read_vnnlib_simple(filename, vars_in, vars_out)

