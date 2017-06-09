#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>
#include <limits>
#include <cmath>

#include <face_quda.h>

#include <iostream>

#ifdef BLOCKSOLVER
#include <Eigen/Dense>
#endif


namespace quda {

  Cheby::Cheby(DiracMatrix &_mat, ChebyParam &_param, TimeProfile &_profile) :
    mat(mat), param(param),profile(profile),init(false) {
  }

  Cheby::~Cheby() {
    if ( init ) {
      for (auto pi : p) delete pi;
      delete rp;
      delete yp;
      delete App;
      delete tmpp;
      init = false;
    }
  }


  ChebyNR::ChebyNR(DiracMatrix &mat, ChebyParam &param, TimeProfile &profile) :
    Cheby(mdagm, param, profile), mdagm(mat.Expose()), init(false) {
  }

  ChebyNR::~ChebyNR() {
    if ( init ) {
      delete bp;
      init = false;
    }
  }

  // CGNR: Mdag M x = Mdag b is solved.
  void ChebyNR::operator()(ColorSpinorField &x, ColorSpinorField &b) {
//    const int iter0 = param.iter;

    if (!init) {
      ColorSpinorParam csParam(b);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      bp = ColorSpinorField::Create(csParam);

      init = true;

    }

    Cheby::operator()(x,*bp);

  }

  void Cheby::operator()(ColorSpinorField &x, ColorSpinorField &b) {
    if (Location(x, b) != QUDA_CUDA_FIELD_LOCATION)
      errorQuda("Not supported");

    profile.TPSTART(QUDA_PROFILE_INIT);

    ColorSpinorParam csParam(x);
    if (!init) {

      csParam.create = QUDA_NULL_FIELD_CREATE;
      rp = ColorSpinorField::Create(b, csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = ColorSpinorField::Create(b, csParam);
      App = ColorSpinorField::Create(b, csParam);
      tmpp = ColorSpinorField::Create(b, csParam);
      init = true;

    }
    ColorSpinorField *T0 = rp;
    ColorSpinorField *T1 = yp;
    ColorSpinorField *T2 = App;
    ColorSpinorField *Tnm = T0;
    ColorSpinorField *Tn = T1;
    ColorSpinorField *Tnp = T2;
    ColorSpinorField &tmp = *tmpp;
    ColorSpinorField *yp =  ColorSpinorField::Create(b, csParam);
    ColorSpinorField &y = *yp;

    double xscale = 2.0/(param.hi-param.lo);
    double mscale = -(param.hi+param.lo)/(param.hi-param.lo);
    mat(*T1, *T0, tmp); // r= Ax
    blas::axpby(mscale,*T0,xscale,*T1);
    blas::axpby(-1,*T0,2.,*T1);
    


    for(int i =2;i<param.order;i++){
      mat(*Tnp, *Tn, tmp); // r= Ax
      blas::axpby(mscale,*Tn,xscale,*Tnp);
      blas::axpby(-1,*Tn,2.,*Tnp);
      ColorSpinorField *swizzle = Tnm;
      Tnm = Tn;
      Tn = Tnp;
      Tnp = swizzle;
    }
    blas::copy(x,*Tn);


    // reset the flops counters
    blas::flops = 0;
    mat.flops();

    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }


}  // namespace quda
