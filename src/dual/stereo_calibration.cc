#include "stereo_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <vector>

namespace dual {
namespace {
std::vector<double> numbers(const std::string& text) {
  static const std::regex re(R"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)");
  std::vector<double> out;
  for(std::sregex_iterator i(text.begin(),text.end(),re),e;i!=e;++i)
    out.push_back(std::stod(i->str()));
  return out;
}
bool field(const std::string& text,const std::string& key,size_t count,
           std::vector<double>* out) {
  auto p=text.find(key+":"); if(p==std::string::npos) return false;
  auto b=text.find('[',p); auto e=text.find(']',b);
  if(b==std::string::npos||e==std::string::npos) return false;
  *out=numbers(text.substr(b+1,e-b-1)); return out->size()==count;
}
Point3f mul(const std::array<double,9>& m,Point3f p) {
  return {m[0]*p.x+m[1]*p.y+m[2]*p.z,
          m[3]*p.x+m[4]*p.y+m[5]*p.z,
          m[6]*p.x+m[7]*p.y+m[8]*p.z};
}
double dot(Point3f a,Point3f b){return a.x*b.x+a.y*b.y+a.z*b.z;}
Point3f add(Point3f a,Point3f b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
Point3f scale(Point3f a,double s){return {a.x*s,a.y*s,a.z*s};}
}

bool StereoCalibration::load(const std::string& path,std::string* error) {
  valid_=false;homography_only_=false; std::ifstream in(path);
  if(!in){if(error)*error="cannot open calibration: "+path;return false;}
  std::ostringstream s;s<<in.rdbuf();const std::string text=s.str();
  std::vector<double> v;
  auto get=[&](const char* k,size_t n){return field(text,k,n,&v);};
  if(!get("image_size",2)){if(error)*error="missing image_size[2]";return false;}
  width_=static_cast<int>(v[0]);height_=static_cast<int>(v[1]);
  if(get("homography_H",9)) {
    std::copy(v.begin(),v.end(),h_.begin());
    double det=h_[0]*(h_[4]*h_[8]-h_[5]*h_[7])-
               h_[1]*(h_[3]*h_[8]-h_[5]*h_[6])+
               h_[2]*(h_[3]*h_[7]-h_[4]*h_[6]);
    if(width_<=0||height_<=0||std::abs(det)<1e-12) {
      if(error)*error="invalid homography calibration";return false;
    }
    homography_only_=true;valid_=true;return true;
  }
  if(!get("camera0_K",9)){if(error)*error="missing camera0_K[9]";return false;}
  std::copy(v.begin(),v.end(),cameras_[0].k.begin());
  if(!get("camera0_D",5)){if(error)*error="missing camera0_D[5]";return false;}
  std::copy(v.begin(),v.end(),cameras_[0].d.begin());
  if(!get("camera1_K",9)){if(error)*error="missing camera1_K[9]";return false;}
  std::copy(v.begin(),v.end(),cameras_[1].k.begin());
  if(!get("camera1_D",5)){if(error)*error="missing camera1_D[5]";return false;}
  std::copy(v.begin(),v.end(),cameras_[1].d.begin());
  if(!get("stereo_R",9)){if(error)*error="missing stereo_R[9]";return false;}
  std::copy(v.begin(),v.end(),r_.begin());
  if(!get("stereo_T",3)){if(error)*error="missing stereo_T[3]";return false;}
  std::copy(v.begin(),v.end(),t_.begin());
  if(!get("stereo_F",9)){if(error)*error="missing stereo_F[9]";return false;}
  std::copy(v.begin(),v.end(),f_.begin());
  if(get("working_distance_m",2)){min_depth_m_=v[0];max_depth_m_=v[1];}
  if(width_<=0||height_<=0||cameras_[0].k[0]<=0||cameras_[1].k[0]<=0||
     min_depth_m_<=0||max_depth_m_<=min_depth_m_) {
    if(error)*error="invalid calibration values";return false;
  }
  valid_=true;return true;
}

Point2f StereoCalibration::map_homography(Point2f p) const {
  double w=h_[6]*p.x+h_[7]*p.y+h_[8];if(std::abs(w)<1e-12)return{-1,-1};
  return{(h_[0]*p.x+h_[1]*p.y+h_[2])/w,
         (h_[3]*p.x+h_[4]*p.y+h_[5])/w};
}

Point2f StereoCalibration::undistort_normalized(int camera,Point2f p) const {
  const auto& c=cameras_[camera?1:0];
  double xd=(p.x-c.k[2])/c.k[0], yd=(p.y-c.k[5])/c.k[4];
  double x=xd,y=yd;
  for(int i=0;i<8;++i){
    double r2=x*x+y*y,r4=r2*r2,r6=r4*r2;
    double radial=1+c.d[0]*r2+c.d[1]*r4+c.d[4]*r6;
    double dx=2*c.d[2]*x*y+c.d[3]*(r2+2*x*x);
    double dy=c.d[2]*(r2+2*y*y)+2*c.d[3]*x*y;
    x=(xd-dx)/radial;y=(yd-dy)/radial;
  }
  return {x,y};
}

Point2f StereoCalibration::distort_pixel(const CameraCalibration& c,Point2f p) const {
  double r2=p.x*p.x+p.y*p.y,r4=r2*r2,r6=r4*r2;
  double radial=1+c.d[0]*r2+c.d[1]*r4+c.d[4]*r6;
  double x=p.x*radial+2*c.d[2]*p.x*p.y+c.d[3]*(r2+2*p.x*p.x);
  double y=p.y*radial+c.d[2]*(r2+2*p.y*p.y)+2*c.d[3]*p.x*p.y;
  return {c.k[0]*x+c.k[2],c.k[4]*y+c.k[5]};
}

Point2f StereoCalibration::project_to_camera1(Point2f p,double depth) const {
  Point2f n=undistort_normalized(0,p);
  Point3f q=add(mul(r_,{n.x*depth,n.y*depth,depth}),{t_[0],t_[1],t_[2]});
  if(q.z<=1e-9)return {-1,-1};
  return distort_pixel(cameras_[1],{q.x/q.z,q.y/q.z});
}

double StereoCalibration::epipolar_distance(Point2f a,Point2f b) const {
  double lx=f_[0]*a.x+f_[1]*a.y+f_[2];
  double ly=f_[3]*a.x+f_[4]*a.y+f_[5];
  double lz=f_[6]*a.x+f_[7]*a.y+f_[8];
  return std::abs(lx*b.x+ly*b.y+lz)/std::max(1e-12,std::hypot(lx,ly));
}

double StereoCalibration::triangulate_depth(Point2f a,Point2f b) const {
  Point2f n0=undistort_normalized(0,a),n1=undistort_normalized(1,b);
  Point3f d0{n0.x,n0.y,1}, d1c1{n1.x,n1.y,1};
  // camera-1 centre and ray expressed in camera-0 coordinates: C=-R^T T.
  std::array<double,9> rt{r_[0],r_[3],r_[6],r_[1],r_[4],r_[7],r_[2],r_[5],r_[8]};
  Point3f d1=mul(rt,d1c1), c1=scale(mul(rt,{t_[0],t_[1],t_[2]}),-1);
  double aa=dot(d0,d0),bb=dot(d0,d1),cc=dot(d1,d1);
  double rhs0=dot(d0,c1),rhs1=dot(d1,c1),det=aa*cc-bb*bb;
  if(std::abs(det)<1e-12)return -1;
  double lambda=(rhs0*cc-bb*rhs1)/det;
  return lambda>0?lambda:-1;
}
} // namespace dual
