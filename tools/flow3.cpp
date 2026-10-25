// 分步效果图：每步一张，旁边标注"处理方法 / 目标 / 实测"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif
#include "chardet.hpp"
#include "common/console_utf8.hpp"
namespace fs=std::filesystem;
namespace{
std::string w2u(const std::wstring& w){if(w.empty())return{};int n=::WideCharToMultiByte(CP_UTF8,0,w.data(),(int)w.size(),nullptr,0,nullptr,nullptr);std::string o(n,'\0');::WideCharToMultiByte(CP_UTF8,0,w.data(),(int)w.size(),o.data(),n,nullptr,nullptr);return o;}
fs::path u8p(const std::string& s){return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()),s.size()));}
bool rd(const std::string& p,cv::Mat& o){std::ifstream f(u8p(p),std::ios::binary);if(!f)return false;f.seekg(0,std::ios::end);auto n=f.tellg();f.seekg(0,std::ios::beg);if(n<=0)return false;std::vector<uchar> b((size_t)n);f.read((char*)b.data(),n);if(!f)return false;o=cv::imdecode(b,cv::IMREAD_GRAYSCALE);return !o.empty();}
cv::Mat g3(const cv::Mat& m){cv::Mat o;if(m.channels()==1)cv::cvtColor(m,o,cv::COLOR_GRAY2BGR);else o=m.clone();return o;}
// 右侧信息面板
cv::Mat panel(const std::string& no,const std::string& name,const std::vector<std::string>& lines,int W,int H){
    cv::Mat p=cv::Mat::zeros(H,W,CV_8UC3);
    cv::putText(p,no,cv::Point(14,40),cv::FONT_HERSHEY_SIMPLEX,1.0,cv::Scalar(0,200,255),2,cv::LINE_AA);
    cv::putText(p,name,cv::Point(14,78),cv::FONT_HERSHEY_SIMPLEX,0.62,cv::Scalar(255,255,255),2,cv::LINE_AA);
    cv::line(p,cv::Point(14,92),cv::Point(W-14,92),cv::Scalar(80,80,80),1);
    int y=120;
    for(const auto& ln:lines){
        cv::Scalar c(200,200,200);
        if(ln.rfind("[",0)==0){c=cv::Scalar(120,220,120);}
        if(ln.rfind("!",0)==0){c=cv::Scalar(120,120,255);}
        std::string t=ln;
        if(t.rfind("[",0)==0||t.rfind("!",0)==0)t=t.substr(1);
        cv::putText(p,t,cv::Point(14,y),cv::FONT_HERSHEY_SIMPLEX,0.42,c,1,cv::LINE_AA);
        y+=23;
    }
    return p;
}
cv::Mat imgPane(const cv::Mat& img,int W,int H){
    cv::Mat b=g3(img);
    double s=std::min((W-10.0)/b.cols,(H-10.0)/b.rows);
    cv::Mat r; cv::resize(b,r,cv::Size(),s,s);
    cv::Mat o=cv::Mat::zeros(H,W,CV_8UC3);
    r.copyTo(o(cv::Rect(5,5,std::min(r.cols,W-10),std::min(r.rows,H-10))));
    return o;
}
cv::Mat pair(const cv::Mat& left,const cv::Mat& right,int H=340,int LW=430,int RW=470){
    cv::Mat a=imgPane(left,LW,H), b=imgPane(right,RW,H);
    cv::Mat o; cv::hconcat(std::vector<cv::Mat>{a,b},o); return o;
}
}
int main(){
    markloc::enableUtf8Console();
    int wc=0; LPWSTR* wv=::CommandLineToArgvW(::GetCommandLineW(),&wc);
    std::vector<std::string> a; if(wv){for(int i=0;i<wc;++i)a.push_back(w2u(wv[i]));::LocalFree(wv);}
    if(a.size()<4){std::printf("usage: flow3 <model> <img> <outdir>\n");return 2;}
    chardet::Model cm; if(!chardet::loadModel(a[1],cm))return 1;
    cv::Mat gray; if(!rd(a[2],gray))return 1;
    chardet::Config cfg=cm.config;
    std::string od=a[3];
    char buf[128];

    // ---- 01 原图
    std::vector<cv::Mat> rows;
    rows.push_back(pair(gray,gray));
    // 覆盖第一对：左原图，右塑封体裁剪
    cv::Rect body(300,225,1065,915);
    {
        cv::Mat r1; cv::hconcat(std::vector<cv::Mat>{imgPane(gray,430,340),imgPane(gray(body),430,340)},r1);
        rows[0]=r1;
    }
    std::vector<cv::Mat> pn;
    pn.push_back(panel("01","RAW INPUT",{
        "处理: 读入 tif / 裁到塑封体",
        "[imdecode 读入，避开中文路径问题",
        "[裁剪 x300 y225 1065x915，去掉引脚",
        "目标: 只保留与刻印相关的区域",
        "!引脚、基板与字符检测无关",
        "!全图处理会被引脚高亮干扰"
    },470,340));

    // ---- 02 笔画信号
    cv::Mat sig=chardet::strokeSignal(gray,cv::Rect(0,0,gray.cols,gray.rows),cfg,false);
    cv::Mat sigVis; cv::normalize(sig,sigVis,0,255,cv::NORM_MINMAX,CV_8U);
    {
        cv::Mat r; cv::hconcat(std::vector<cv::Mat>{imgPane(gray(body),430,340),imgPane(sigVis(body),430,340)},r);
        rows.push_back(r);
    }
    pn.push_back(panel("02","STROKE SIGNAL",{
        "处理: 高斯背景估计 + 相减",
        "[GaussianBlur sigma=15 估背景",
        "[原图 - 背景；负值截断",
        "[归一化 q99.5*0.55 + q90*0.45",
        "目标: 让刻印笔画从背景中凸显",
        "!原图笔画与背景仅差约 20 灰度",
        "!sigma 必须远大于笔画宽度(10px)"
    },470,340));

    // ---- 03 二值化
    cv::Mat bin; cv::compare(sig,cfg.strokeHi,bin,cv::CMP_GT);
    {
        cv::Mat r; cv::hconcat(std::vector<cv::Mat>{imgPane(sigVis(body),430,340),imgPane(bin(body),430,340)},r);
        rows.push_back(r);
    }
    pn.push_back(panel("03","BINARY INK",{
        "处理: 阈值二值化",
        "[compare(S > 0.35, CMP_GT)",
        "[0.35 = strokeHi，标定得来",
        "目标: 得到'哪里有墨'的掩码",
        "!标定实验: 阈值0.35时待检墨量",
        "!= 模型墨量的 96%，量纲一致"
    },470,340));

    // ---- 04 模型
    {
        cv::Mat r; cv::hconcat(std::vector<cv::Mat>{imgPane(bin(body),430,340),imgPane(cm.freq(body),430,340)},r);
        rows.push_back(r);
    }
    pn.push_back(panel("04","MODEL FREQ MAP",{
        "处理: 84 张 OK 样本平均",
        "[整图粗对齐 + 逐字符精对齐",
        "[二值累加 / N + 转 0~255",
        "目标: 得到'每个像素有墨的概率'",
        "[freq>0.50 => 必然有墨",
        "[freq<0.15 => 应为空白",
        "!用二值累加，强度累加会使边缘变糊"
    },470,340));

    // ---- 05 字符框
    std::vector<chardet::CharUnit> all;
    for(size_t ri=0;ri<cm.regions.size();++ri){auto u=chardet::toUnits(cm,(int)ri);for(auto&x:u)all.push_back(x);}
    cv::Mat boxed=gray.clone(), boxM=g3(cm.freq);
    cv::cvtColor(boxed,boxed,cv::COLOR_GRAY2BGR);
    {
        for(size_t ri=0;ri<cm.regions.size();++ri)cv::rectangle(boxM,cm.regions[ri].rect,cv::Scalar(255,80,0),2);
        for(auto&u:all){cv::rectangle(boxM,u.bbox,cv::Scalar(0,255,0),1);}
        cv::Mat r; cv::hconcat(std::vector<cv::Mat>{imgPane(boxM(body),430,340),imgPane(boxed(body),430,340)},r);
        rows.push_back(r);
    }
    snprintf(buf,sizeof(buf),"%zu",all.size());
    pn.push_back(panel("05","CHAR BOXES (split)",{
        "处理: 列投影切分字符",
        "[freq 二值化 (binFreq=0.25)",
        "[列投影，阈值=峰值x0.03",
        "[间隙<中位字宽x0.10 则合并",
        "目标: 把一行字切成单字框",
        "!合并间隙解决 i 的点与竖被切开",
        std::string("[共 ")+buf+" 个字符"
    },470,340));

    // ---- 06 整图配准
    cv::Mat sigReg=sig.clone();
    cv::Point2d sh(0,0);
    if(!cm.ref.empty()){
        cv::Mat r32,s32; cm.ref.convertTo(r32,CV_32F); sig.convertTo(s32,CV_32F);
        cv::Rect ia;
        for(auto&u:all)ia=ia.area()>0?(ia|u.bbox):u.bbox;
        ia.x=std::max(0,ia.x-20);ia.y=std::max(0,ia.y-20);
        ia.width=std::min(sig.cols-ia.x,ia.width+40);ia.height=std::min(sig.rows-ia.y,ia.height+40);
        cv::Mat rC,sC; r32(ia).copyTo(rC); s32(ia).copyTo(sC);
        sh=cv::phaseCorrelate(sC,rC,cv::noArray());
        if(std::abs(sh.x)<60&&std::abs(sh.y)<60){
            cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sh.x,0,1,-sh.y);
            cv::warpAffine(sig,sigReg,M,sig.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);
        }
    }
    cv::Mat regVis; cv::normalize(sigReg,regVis,0,255,cv::NORM_MINMAX,CV_8U);
    {
        cv::Mat r; cv::hconcat(std::vector<cv::Mat>{imgPane(sigVis(body),430,340),imgPane(regVis(body),430,340)},r);
        rows.push_back(r);
    }
    snprintf(buf,sizeof(buf),"(%+.1f, %+.1f)",-sh.x,-sh.y);
    pn.push_back(panel("06","REGISTER",{
        "处理: 相位相关配准",
        "[只对'字符区域'做，不用全图",
        "[phaseCorrelate + 亚像素插值",
        std::string("平移量 ")+buf,
        "目标: 消除整体位移",
        "!全图会被引脚高亮主导",
        "!实测误差可达 8~16px"
    },470,340));

    // ---- 07 逐字符对齐
    cv::Mat aligned=g3(gray);
    int moved=0;
    for(auto&u:all){
        cv::Rect box=u.bbox&cv::Rect(0,0,sigReg.cols,sigReg.rows);
        if(box.width<=4||box.height<=4)continue;
        constexpr int KS=12;
        cv::Rect sr(box.x-KS,box.y-KS,box.width+2*KS,box.height+2*KS);
        sr&=cv::Rect(0,0,sigReg.cols,sigReg.rows);
        if(sr.width<box.width||sr.height<box.height)continue;
        cv::Mat mS,tS; cm.freq(sr).convertTo(mS,CV_32F,1.0/255.0); sigReg(sr).convertTo(tS,CV_32F);
        cv::Mat res; cv::matchTemplate(tS,mS(cv::Rect(box.x-sr.x,box.y-sr.y,box.width,box.height)),res,cv::TM_CCOEFF_NORMED);
        cv::Point mL; double mv=0; cv::minMaxLoc(res,nullptr,&mv,nullptr,&mL);
        if(mv<0.30)continue;
        cv::Rect nb(box.x+mL.x-KS,box.y+mL.y-KS,box.width,box.height);
        if((nb&cv::Rect(0,0,sigReg.cols,sigReg.rows))!=nb)continue;
        cv::rectangle(aligned,u.bbox,cv::Scalar(0,0,255),1);
        cv::rectangle(aligned,nb,cv::Scalar(0,255,0),2);
        u.bbox=nb; ++moved;
    }
    {
        cv::Mat r; cv::hconcat(std::vector<cv::Mat>{imgPane(boxed(body),430,340),imgPane(aligned(body),430,340)},r);
        rows.push_back(r);
    }
    snprintf(buf,sizeof(buf),"%d / %zu",moved,all.size());
    pn.push_back(panel("07","PER-CHAR ALIGN",{
        "处理: 每个字单独模板匹配",
        "[搜索范围 +-12px",
        "[抛物线插值到亚像素",
        "[红=原框  绿=对齐后",
        std::string("移动 ")+buf+" 个框",
        "目标: 消除每字 6~8px 的个体错位",
        "!不对齐时框内装'半个字+空白'"
    },470,340));

    // ---- 08 检测
    cv::Mat det=g3(gray);
    int nm=0,ne=0;
    for(auto&u:all){
        auto nc=chardet::diffNcc(sigReg,cm.freq,u,cfg);
        if(!nc.valid)continue;
        if(nc.missScore>0)   {cv::rectangle(det,u.bbox,cv::Scalar(0,0,255),2);++nm;}
        if(nc.extraScore>0)  {cv::rectangle(det,u.bbox,cv::Scalar(255,120,0),2);++ne;}
    }
    auto bd=chardet::diffBlocks(sigReg,cm.freq,all,cfg,8);
    for(auto&b:bd.extraBlocks)cv::rectangle(det,cv::Rect(b.x0,b.y0,b.x1-b.x0,b.y1-b.y0),cv::Scalar(0,220,220),2);
    {
        cv::Mat r; cv::hconcat(std::vector<cv::Mat>{imgPane(aligned(body),430,340),imgPane(det(body),430,340)},r);
        rows.push_back(r);
    }
    snprintf(buf,sizeof(buf),"miss=%d extra=%d blk=%d",nm,ne,bd.extraCount);
    pn.push_back(panel("08","DETECT",{
        "处理: 多判据并行",
        "[读句1 字符级: 必有墨处缺墨?",
        "[读句2 整行: 字符区间有无缺字?",
        "[读句3 划痕: Hough直线+灰度分级",
        "[读句4 污损/溢料: 仅字符框内",
        std::string("结果 ")+buf,
        "!任一命中 => NG"
    },470,340));

    // ---- 拼图
    const int PH=340,PW=470;
    cv::Mat out;
    std::vector<cv::Mat> fin;
    for(size_t i=0;i<rows.size();++i){
        cv::Mat r; cv::hconcat(rows[i],pn[i],r);
        // 各列宽度已一致，无需补边
        fin.push_back(r);
    }
    cv::vconcat(fin,out);
    // 加分隔线
    for(int i=1;i<(int)fin.size();++i){
        cv::line(out,cv::Point(0,i*PH),cv::Point(out.cols,i*PH),cv::Scalar(60,60,60),1);
    }
    cv::imwrite(od,out);
    std::printf("saved %s (%dx%d)\n",od.c_str(),out.cols,out.rows);
    return 0;
}
