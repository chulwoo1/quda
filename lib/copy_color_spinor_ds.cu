#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorDS(ColorSpinorField &dst, const ColorSpinorField &src, 
				QudaFieldLocation location, void *Dst, void *Src, 
				void *dstNorm, void *srcNorm) {
    CopyGenericColorSpinor<3>(dst, src, location, (double*)Dst, (float*)Src);
  }  

} // namespace quda
