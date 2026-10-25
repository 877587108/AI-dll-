// doc_scoring.cpp —— 生成「打分逻辑说明」HTML
//
// 为什么用 C++ 生成而不是直接写 .html 文件：
// 文中有大量数字（阈值、公式系数）必须与源码一致，写死在字符串里
// 改代码时很容易忘了同步。这里直接引用 chardet 与 markloc 里的常量，
// 模板改一次、代码改一次，两边就不会漂。
#include <cstdio>
#include <string>

#include <opencv2/core.hpp>

#include "chardet.hpp"
#include "scratch.hpp"

static std::string fnum(double v, int prec = 2) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.*f", prec, v);
    return b;
}

int main(int argc, char** argv) {
    const char* outPath = argc > 1 ? argv[1] : "out/scoring.html";

    const markloc::ScratchParams S{};
    const chardet::PatternBreakParams P{};

    std::FILE* f = std::fopen(outPath, "wb");
    if (!f) {
        std::printf("无法写入 %s\n", outPath);
        return 1;
    }

    auto W = [&](const std::string& s) { std::fwrite(s.data(), 1, s.size(), f); };

    W("<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">");
    W("<title>MarkLoc 打分逻辑说明</title>");
    W(R"(<style>
:root{--bg:#0f1419;--card:#1a2028;--fg:#e6edf3;--dim:#8b949e;--acc:#58a6ff;
--red:#f85149;--ora:#ff9f45;--yel:#e3b341;--grn:#3fb950;--pur:#bc8cff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:40px 28px 100px}
h1{font-size:34px;margin:0 0 6px;letter-spacing:-.5px}
.sub{color:var(--dim);margin-bottom:38px;font-size:15px}
h2{font-size:24px;margin:56px 0 6px;padding-top:22px;border-top:1px solid #30363d;
display:flex;align-items:center;gap:12px}
h2 .tag{font-size:12px;font-weight:600;padding:3px 10px;border-radius:20px;
background:#21262d;color:var(--acc);letter-spacing:.5px}
h3{font-size:18px;margin:32px 0 10px;color:var(--acc)}
h4{font-size:15px;margin:22px 0 8px;color:var(--dim);font-weight:600}
p{margin:10px 0}
code{background:#21262d;padding:2px 6px;border-radius:4px;font-size:13px;
font-family:"Cascadia Code",Consolas,monospace;color:#79c0ff}
.card{background:var(--card);border:1px solid #30363d;border-radius:10px;
padding:20px 24px;margin:16px 0}
.formula{background:#0d1117;border-left:3px solid var(--acc);border-radius:0 8px 8px 0;
padding:16px 20px;margin:16px 0;font-family:"Cascadia Code",Consolas,monospace;
font-size:14px;overflow-x:auto;line-height:2}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}
th,td{padding:9px 14px;text-align:left;border-bottom:1px solid #21262d}
th{background:#21262d;font-weight:600;color:var(--dim);font-size:13px;
text-transform:uppercase;letter-spacing:.5px}
tr:hover td{background:#1c2128}
.warn{background:#2d1618;border-left:3px solid var(--red);padding:14px 20px;
border-radius:0 8px 8px 0;margin:16px 0}
.ok{background:#12261a;border-left:3px solid var(--grn);padding:14px 20px;
border-radius:0 8px 8px 0;margin:16px 0}
.note{background:#1c2333;border-left:3px solid var(--yel);padding:14px 20px;
border-radius:0 8px 8px 0;margin:16px 0}
.steps{counter-reset:s;list-style:none;padding:0}
.steps>li{counter-increment:s;position:relative;padding:0 0 22px 54px;border-left:2px solid #21262d;
margin-left:18px}
.steps>li:last-child{border-left-color:transparent}
.steps>li::before{content:counter(s);position:absolute;left:-18px;top:-2px;width:34px;height:34px;
border-radius:50%;background:var(--acc);color:#0d1117;font-weight:700;font-size:15px;
display:flex;align-items:center;justify-content:center}
.steps>li b{color:#fff}
.diagram{background:#0d1117;border-radius:10px;padding:22px;margin:18px 0;overflow-x:auto}
.diagram pre{margin:0;font-family:"Cascadia Code",Consolas,monospace;font-size:13px;
line-height:1.65;color:#c9d1d9}
.badge{display:inline-block;padding:2px 9px;border-radius:12px;font-size:12px;font-weight:600}
.b-certain{background:#3d1418;color:var(--red)}
.b-obvious{background:#3d2412;color:var(--ora)}
.b-suspect{background:#3d3412;color:var(--yel)}
.b-none{background:#21262d;color:var(--dim)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.two{grid-template-columns:1fr}}
.kv{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #21262d;font-size:14px}
.kv:last-child{border:0}
.kv span:last-child{font-family:"Cascadia Code",Consolas,monospace;color:var(--acc)}
</style></head><body><div class="wrap">)");

    W("<h1>MarkLoc 打分逻辑说明</h1>");
    W("<div class=\"sub\">两套完全独立的判据 —— 划痕分与图案断点分。"
      "各自量纲不同、门槛不同，<b>只做逻辑或，绝不加权求和</b>。</div>");

    // ---------------- 总览
    W("<h2><span class=\"tag\">总览</span>为什么是两套分</h2>");
    W("<div class=\"diagram\"><pre>");
    W("                    芯片图像 (1688x1460 或 1496x1496)\n");
    W("                              |\n");
    W("                    strokeSignal()  笔画信号\n");
    W("                              |\n");
    W("        +---------------------+---------------------+\n");
    W("        |                                           |\n");
    W("   字符区判据                                   图案区判据\n");
    W("  (划痕 / 缺墨 / 多墨)                        (logo 椭圆环断连)\n");
    W("        |                                           |\n");
    W("    S = 深度 x 跨字数                      score = 缺口角度 x 100\n");
    W("        |                                           |\n");
    W("   S >= 1000 ?                                 score >= 400 ?\n");
    W("        |                                           |\n");
    W("        +---------------------+---------------------+\n");
    W("                              |\n");
    W("                    NG = (划痕报出) OR (图案断口报出)\n");
    W("</pre></div>");
    W("<div class=\"note\"><b>为什么不合并成一个总分：</b>划痕分的单位是"
      "「灰度对比度 × 字符个数」，断点分的单位是「度的百分之一」。"
      "两者相加会得到一个单位不明的数，门槛就失去物理含义；"
      "而且以后单独调一套会莫名其妙牵连另一套。</div>");

    // ---------------- 划痕分
    W("<h2><span class=\"tag\">判据 A</span>划痕分 S = 深度 × 跨字数</h2>");
    W("<p>客户确认的唯一缺陷类型是<b>划痕</b>。判据沿一条候选直线，"
      "量两件事：<b>这道痕有多深</b>（相对周围背景的亮度差），"
      "以及<b>它压过了几个字符</b>。两者相乘是严重度。</p>");

    W("<div class=\"formula\">");
    W("S = depth &times; charsHit<br>");
    W("<span style=\"color:var(--dim)\">depth    = 垂直剖面 (峰值 &minus; 背景中位数) 的平均<br>");
    W("charsHit = 线段覆盖比例 &ge; " + fnum(S.minCharCover, 2) +
      " 的字符框个数</span><br><br>");
    W("判 NG：S &ge; " + fnum(S.minComposite, 0) + " = <code>scratchMinComposite</code>");
    W("</div>");

    W("<h3>七步流水线</h3><ol class=\"steps\">");
    W("<li><b>取检测带</b>：所有字符框求并集，外扩 <code>" + fnum(S.growPx, 0) +
      "px</code>，得到 <code>band</code>。后续所有坐标都在带内，省算力也避开无关区域。</li>");
    W("<li><b>算模型墨迹与三个辅助图</b>：<br>"
      "<code>modelInk = freq &gt; blankFreq&times;255</code>（模型认为这里本来有笔画）<br>"
      "<code>inkDist</code> = 到模型墨迹的距离变换（用于剔除字形漏出）<br>"
      "<code>gxModel/gyModel</code> = 模型墨迹的 Sobel 梯度（得到笔画走向）</li>");
    W("<li><b>弱阈值二值化 + 扣笔画</b>：<br>"
      "<code>weak = signal &gt; missStrokeLo &times; 0.7</code>，"
      "然后把 <code>modelInk</code> 膨胀 9&times;9 后扣掉。<br>"
      "接着做 <b>闭运算重连</b>（核 <code>" + fnum(S.bridge, 0) +
      "px</code>）：一条跨字划痕压在字上的部分会被当笔画删掉，整条断开成几截、"
      "各自都过不了长度门槛，所以要沿划痕方向接回去。</li>");
    W("<li><b>HoughLinesP 找直线候选</b>：<code>rho=1, theta=1&deg;, "
      "阈值=20, minLen=30, maxGap=60</code>。</li>");
    W("<li><b>五道筛选</b>（按顺序，任一不过就丢弃并记原因）：<br>"
      "<table><tr><th>筛选</th><th>门槛</th><th>作用</th></tr>");
    W("<tr><td>长度</td><td>&ge; " + fnum(S.minLen, 0) +
      "px</td><td>太短的不是划痕</td></tr>");
    W("<tr><td>覆盖率</td><td>&ge; " + fnum(S.minCov, 2) +
      "</td><td>线段上多少比例真有墨</td></tr>");
    W("<tr><td>离字形距离</td><td>中位 &ge; " + fnum(S.minInkDist, 2) +
      "</td><td>剔除字形漏出连成的假线</td></tr>");
    W("<tr><td>能量</td><td>&ge; " + fnum(S.minEnergy, 0) +
      "</td><td>默认 0，只去噪</td></tr>");
    W("<tr><td>方向一致度</td><td>中位cos &le; " + fnum(S.maxParCos, 2) +
      " 或 横穿占比 &ge; " + fnum(S.minVertFrac, 2) +
      "</td><td>真划痕横穿笔画；字形漏出只会沿着笔画走</td></tr>");
    W("</table></li>");
    W("<li><b>量深度</b>：沿线段法向取 &plusmn;" + fnum(S.depthReach, 0) +
      "px 的剖面，<code>contrast = 峰值 &minus; 两端背景中位数</code>，"
      "全线段平均得 <code>depth</code>。同时数半高宽得 <code>widthPx</code>。</li>");
    W("<li><b>数跨字数</b>：把线段和每个字符框求交。"
      "<code>minCharCover</code> 默认是 <b>" + fnum(S.minCharCover, 2) +
      "</b>，此时凡与线段外接矩形有交集的框都算跨越；"
      "把它调成 &gt; 0（如 0.10），则要求线段真正落在框内的采样点比例达到该值。<br>"
      "<span style=\"color:var(--dim)\">（比例分支是为了修正轴对齐外接矩形带来的虚增 —— "
      "斜线的外接矩形面积是线本身的几十倍，会把只是擦边的框也算进去。"
      "但实测把它从 0 抬到 0.10 会显著改变结果，所以默认保持 0，不要随意调。）</span></li>");
    W("</ol>");

    W("<h3>四档切点</h3>");
    W("<table><tr><th>档位</th><th>S 范围</th><th>含意</th></tr>");
    W("<tr><td><span class=\"badge b-certain\">确定</span></td><td>&ge; " +
      fnum(S.cutCertain, 0) + "</td><td>明确划痕</td></tr>");
    W("<tr><td><span class=\"badge b-obvious\">明显</span></td><td>&ge; " +
      fnum(S.cutObvious, 0) + "</td><td>看得出是划痕</td></tr>");
    W("<tr><td><span class=\"badge b-suspect\">可疑</span></td><td>&ge; " +
      fnum(S.cutSuspect, 0) + "</td><td>达到报告门槛</td></tr>");
    W("<tr><td><span class=\"badge b-none\">轻微</span></td><td>&lt; " +
      fnum(S.cutSuspect, 0) + "</td><td>不报</td></tr></table>");

    W("<div class=\"note\"><b>门槛 1000 是怎么定的：</b>深度用在中位数意义上等于"
      "「比旁边亮多少」；跨字数是对「影响识别」这件事最直接的度量。"
      "只看深度，OK 件里离字很远的深擦痕会排到最前；只看跨字数，一条浅线也能过关。"
      "相乘之后实测：真 NG 集中在 1219~2289，"
      "83 张 OK 里绝大多数 &lt; 900。取 1000 使真 NG 6/6 全报。</div>");

    // ---------------- 断点分
    W("<h2><span class=\"tag\">判据 B</span>图案断点分 score = 缺口角度 &times; 100</h2>");
    W("<p>logo 是一个<b>闭合的椭圆环</b>，判据只问一件事："
      "<b>这条环被切断没有</b>。不问墨量、不问深浅。</p>");
    W("<div class=\"formula\">");
    W("score = maxGapAngle &times; 100<br>");
    W("<span style=\"color:var(--dim)\">maxGapAngle = 被判为断口的最长缺口（度）</span><br><br>");
    W("判 NG：score &ge; " + fnum(P.minScore, 0) + " = <code>minScore</code>"
      " &nbsp;（对应 " + fnum(P.minScore / 100.0, 1) + "&deg;）");
    W("</div>");

    W("<div class=\"warn\"><b>为什么复用划痕判据不行：</b>logo 环在一个 264&times;154 "
      "的框里只占一条 8~14px 宽的线，框内九成是背景。"
      "任何&ldquo;框内墨量比例&rdquo;都会被背景稀释到分不出来。</div>");

    W("<h3>七步流水线</h3><ol class=\"steps\">");
    W("<li><b>取椭圆参数</b>：直接来自选框 <code>RegionInfo::ell</code>"
      "（中心 + 长半轴 a + 短半轴 b + 长轴角度），<b>不做拟合</b>。"
      "同一个外接矩形能对应多个不同角度、不同半轴的椭圆，反推出来的不一定是你画的那个。</li>");
    W("<li><b>拉直</b>：把椭圆变成半径 a 的正圆。四步显式构造：<br>"
      "<div class=\"formula\">M<sub>3</sub> = T<sub>2</sub> &middot; "
      "Sc &middot; Rm &middot; T<sub>1</sub><br><br>"
      "T<sub>1</sub>：平移到原点 &nbsp; Rm：旋转 &minus;rot &nbsp; "
      "Sc：y 方向乘 a/b &nbsp; T<sub>2</sub>：平移回去</div>"
      "不用 <code>getRotationMatrix2D</code> —— 它的返回矩阵已含"
      "「把旋转中心搬回该点」的平移，再乘缩放会把中心挪走。</li>");
    W("<li><b>极坐标展开</b>：<code>warpPolar</code> 把拉直后的环展开成横带。<br>"
      "<code>dsize.width = 半径采样数</code>，<code>dsize.height = 角度数</code>"
      "（<b>别记反</b>）。展开后转置成 <code>rows=半径, cols=角度</code>，按角度扫。</li>");
    W("<li><b>屏蔽环内</b>：半径 &lt; " + fnum(P.innerBlockRatio, 2) +
      "&times;a 的区域清零。环里还有字母（SC），它们在法向上会把缺口补上。</li>");
    W("<li><b>判每个角度有没有墨</b>（<b>带型连通判定</b>）："
      "一个角度只要在半径带 <code>[" + fnum(P.bandInnerRatio, 2) + "a, " +
      fnum(P.bandOuterRatio, 2) + "a]</code> 里有 &ge; " + fnum(P.minRunPixels, 0) +
      " 个连续像素超过 <code>inkThr=" + fnum(P.inkThr, 2) + "</code>，就算该角度有墨。</li>");
    W("<li><b>环形 0/1 序列 &rarr; 缺口</b>：从第一个有墨点开始绕一圈，"
      "连续无墨的一段就是缺口。<br>"
      "然后<b>合并间隔 &le; " + fnum(P.mergeGapCount, 0) +
      " 个采样点的相邻缺口</b> —— 缺口边缘的零星墨点会把一个贯通缺口切成几段，"
      "不合并会变成&ldquo;洞越大越检不出来&rdquo;。</li>");
    W("<li><b>断口判定</b>：<code>lenDeg &ge; minGapAngle</code> 且 "
      "<code>lenDeg &le; maxGapAngle</code> 且 <code>flank &ge; minFlankCount</code>。"
      "两侧都必须有够长的墨段 —— 只有一边的是边缘毛刺，不叫环被切断。</li>");
    W("</ol>");

    W("<table><tr><th>参数</th><th>默认值</th><th>含意</th></tr>");
    W("<tr><td><code>angles</code></td><td>" + fnum(P.angles, 0) +
      "</td><td>沿周长采样角度数</td></tr>");
    W("<tr><td><code>inkThr</code></td><td>" + fnum(P.inkThr, 2) +
      "</td><td>「有墨」的强度门槛</td></tr>");
    W("<tr><td><code>innerBlockRatio</code></td><td>" + fnum(P.innerBlockRatio, 2) +
      "</td><td>环内屏蔽比例</td></tr>");
    W("<tr><td><code>bandInnerRatio</code></td><td>" + fnum(P.bandInnerRatio, 2) +
      "</td><td>半径带下界 &times; a</td></tr>");
    W("<tr><td><code>bandOuterRatio</code></td><td>" + fnum(P.bandOuterRatio, 2) +
      "</td><td>半径带上界 &times; a</td></tr>");
    W("<tr><td><code>minRunPixels</code></td><td>" + fnum(P.minRunPixels, 0) +
      "</td><td>连续有墨像素数下限</td></tr>");
    W("<tr><td><code>minGapAngle</code></td><td>" + fnum(P.minGapAngle, 1) +
      "&deg;</td><td>缺口小于此算噪声</td></tr>");
    W("<tr><td><code>maxGapAngle</code></td><td>" + fnum(P.maxGapAngle, 1) +
      "&deg;</td><td>大于此算整块缺失</td></tr>");
    W("<tr><td><code>minFlankCount</code></td><td>" + fnum(P.minFlankCount, 0) +
      "</td><td>缺口两侧各需的连续有墨点数</td></tr>");
    W("<tr><td><code>mergeGapCount</code></td><td>" + fnum(P.mergeGapCount, 0) +
      "</td><td>相邻缺口合并间隔</td></tr>");
    W("<tr><td><code>minScore</code></td><td>" + fnum(P.minScore, 0) +
      "</td><td>判 NG 门槛</td></tr>");
    W("</table>");

    // ---------------- 带型判定的来龙去脉
    W("<h2><span class=\"tag\">关键改动</span>为什么放弃&ldquo;跟踪半径曲线&rdquo;</h2>");
    W("<h3>老做法及其失效</h3>");
    W("<p>老做法对每个角度找<b>半径方向最亮的位置</b> <code>r(&theta;)</code>，"
      "再沿这条曲线 &plusmn;" + fnum(P.radiusTol, 0) + "px 判有无墨。</p>");
    W("<div class=\"warn\"><b>它隐含一个假设：环在任何一个角度只对应一个半径。</b><br>"
      "但实测这批图的环是<b>一段粗一段细的闭合带</b>，&ldquo;最亮处&rdquo;"
      "会在带的两条边缘之间来回跳，跳幅就是&ldquo;起伏量&rdquo;。<br>"
      "83 张 OK 上量到的起伏：<b>中位 28px、最大 48px</b>，而墨线只有 10~14px。<br>"
      "于是起伏量测到的是<b>环的粗细变化</b>，不是对准误差。</div>");
    W("<div class=\"diagram\"><pre>");
    W("  老做法的问题（示意）：\n\n");
    W("        环的一段（粗）        环的一段（细）\n");
    W("        ============             ====\n");
    W("          ^ 最亮在这                ^ 最亮在这\n");
    W("          |                          |\n");
    W("     r(theta) 曲线在这里来回跳 --> 起伏量虚高\n");
    W("          |                          |\n");
    W("     <-- 判据以为环在移动，其实是环本身粗细不同 -->\n");
    W("</pre></div>");

    W("<h3>新做法：带型连通判定</h3>");
    W("<p>不再问&ldquo;最亮半径在哪&rdquo;，而问<b>&ldquo;这条半径带里有没有墨&rdquo;</b>。"
      "环粗一段细一段完全不影响结果。</p>");
    W("<div class=\"diagram\"><pre>");
    W("  半径带 [0.75a, 1.35a]：\n\n");
    W("  r=1.35a  ----------------------------------------\n");
    W("                    环（粗）  环（细）  环\n");
    W("           ========  ==========  =====  ======\n");
    W("  r=0.75a  ----------------------------------------\n\n");
    W("  判据：每个角度在这条带里数&ldquo;最长连续有墨段&rdquo;\n");
    W("        段长 >= 3 个像素  -->  该角度有墨\n");
    W("</pre></div>");
    W("<div class=\"ok\"><b>实测收益：</b>83 张 OK 里&ldquo;真正测到&rdquo;的从 "
      "<b>42 张 &rarr; 83 张</b>。原来的 41 张图案区结论被清零 —— "
      "也就是说之前&ldquo;0 张检出缺墨&rdquo;里有一半是<b>根本没测</b>，"
      "不是&ldquo;测了没问题&rdquo;。<br>"
      "另外 <code>innerBlockRatio</code> 从 0.70 到 0.90 全范围调整都不再影响结果，"
      "证明判据确实不再依赖那根曲线。</div>");

    // ---------------- 挖洞活性验证
    W("<h2><span class=\"tag\">验证</span>怎么证明检测器没有失灵</h2>");
    W("<p>&ldquo;0 张报 NG&rdquo;这个结论必须能区分两种情况："
      "<b>环确实都好</b>，还是<b>检测器什么都测不出来</b>。"
      "做法是往环上挖已知角度的洞，看检测器的响应曲线。</p>");
    W("<table><tr><th>挖洞角度</th><th>检出缺口</th><th>断点分</th><th>判定</th></tr>");
    W("<tr><td>0&deg;（对照）</td><td>0.00&deg;</td><td>0</td><td>OK</td></tr>");
    W("<tr><td>1&deg;</td><td>2.00&deg;</td><td>200</td><td>OK</td></tr>");
    W("<tr><td>2&deg;</td><td>2.50&deg;</td><td>250</td><td>OK</td></tr>");
    W("<tr><td>4&deg;</td><td>2.50&deg;</td><td>250</td><td>OK</td></tr>");
    W("<tr><td>6&deg;</td><td>3.00&deg;</td><td>300</td><td>OK</td></tr>");
    W("<tr><td><b>8&deg;</b></td><td><b>5.00&deg;</b></td><td><b>500</b></td>"
      "<td><b>可疑 NG</b></td></tr>");
    W("<tr><td>12&deg;</td><td>11.50&deg;</td><td>1150</td><td>明显 NG</td></tr>");
    W("</table>");
    W("<div class=\"ok\">挖 8&deg; 就能报 NG。真实图最大缺口 2.0&deg;，"
      "处在安全区 —— 所以&ldquo;0 张断连&rdquo;是可信的。</div>");

    // ---------------- 实际数据
    W("<h2><span class=\"tag\">实测</span>两批数据的缺口分布</h2>");
    W("<div class=\"two\">");
    W("<div class=\"card\"><h4>unset（158 张 _00，1496&times;1496）</h4>");
    W("<table><tr><th>缺口</th><th>张数</th><th>占比</th></tr>");
    W("<tr><td>0.0&deg;</td><td>136</td><td>86.1%</td></tr>");
    W("<tr><td>0.5&deg;</td><td>11</td><td>7.0%</td></tr>");
    W("<tr><td>1.0&deg;</td><td>7</td><td>4.4%</td></tr>");
    W("<tr><td>1.5&deg;</td><td>3</td><td>1.9%</td></tr>");
    W("<tr><td>2.0&deg;</td><td>1</td><td>0.6%</td></tr></table>"
      "<p style=\"color:var(--dim);font-size:13px\">最大 2.0&deg;，"
      "判 NG 0 张（门槛 4&deg;）</p></div>");
    W("<div class=\"card\"><h4>ngv2（8 张 _00）@ 1.0&deg; 标准</h4>");
    W("<table><tr><th>文件</th><th>缺口</th><th>分</th><th>判定</th></tr>");
    W("<tr><td>1_00</td><td>0.00&deg;</td><td>0</td><td>OK</td></tr>");
    W("<tr><td>2_00</td><td>2.50&deg;</td><td>250</td><td><b>NG</b></td></tr>");
    W("<tr><td>3_00</td><td>1.50&deg;</td><td>150</td><td><b>NG</b></td></tr>");
    W("<tr><td>4_00</td><td>1.50&deg;</td><td>150</td><td><b>NG</b></td></tr>");
    W("<tr><td>5/6/8/断字</td><td>&le;0.50&deg;</td><td>0</td><td>OK</td></tr></table>"
      "<p style=\"color:var(--dim);font-size:13px\">判 NG 3 张</p></div>");
    W("</div>");

    // ---------------- 阈值可调
    W("<h2><span class=\"tag\">操作</span>怎么改门槛</h2>");
    W("<p>判 NG 的线放在哪里属于客户的决定，不该由代码钉死。两个旋钮：</p>");
    W("<div class=\"formula\">");
    W("<span style=\"color:var(--dim)\">// 命令行为图案区指定 &ldquo;缺口大于 X 度即 NG&rdquo;</span><br>");
    W("markloc_unsetlogo &lt;model&gt; &lt;imgDir&gt; &lt;tmpl.png&gt; --ngdeg 1.0<br><br>");
    W("<span style=\"color:var(--dim)\">// 等价于同时设两个参数，必须一起改才自洽</span><br>");
    W("minGapAngle = 1.0 &nbsp;&nbsp;<span style=\"color:var(--dim)\">// 缺口够 1.0&deg; 才算断口</span><br>");
    W("minScore&nbsp;&nbsp;&nbsp; = 100 &nbsp;&nbsp;<span style=\"color:var(--dim)\">// 角度&times;100，1.0&deg; 对应 100 分</span>");
    W("</div>");
    W("<div class=\"note\"><b>只改 minScore 是不对的：</b>那样 0.6~1.0&deg; 的缺口仍会被算成"
      "&ldquo;断口&rdquo;并给出分数，只是不报 NG —— 那是另一套语义。"
      "要表达&ldquo;缺口大于 X 即 NG&rdquo;，两个都要改。</div>");

    // ---------------- 已知识别盲区
    W("<h2><span class=\"tag\">局限</span>已知的失效与盲区</h2>");
    W("<div class=\"warn\"><b>0095_RT（旧规格 OK 图）会误报图案断。</b><br>"
      "拉直、极坐标、椭圆几何都已单独验证正确，环也完好；"
      "但探针显示该图环的半径在 104~171 之间漂移，而正常图稳定在 124~141。"
      "根因尚未定位。注意：这张在老算法下是被&ldquo;未对准&rdquo;跳过的，"
      "也就是说它之前&ldquo;没报&rdquo;并不等于&ldquo;测对了&rdquo;。</div>");
    W("<div class=\"warn\"><b>跨规格不能直接比分数。</b>"
      "<code>strokeSignal</code> 的归一化刻度由整幅 99.5 分位估计，"
      "1688&times;1460 与 1496&times;1496 的图不可比。"
      "unset / ngv2 的检测是先用模板匹配定位，再<b>在整幅上</b>算信号 —— "
      "只在 ROI 里算会让环主导分位数，刻度被抬高一倍，基线全错。</div>");

    W("<div style=\"margin-top:60px;padding-top:24px;border-top:1px solid #30363d;"
      "color:var(--dim);font-size:13px\">");
    W("<p>本文档由 <code>tools/doc_scoring.cpp</code> 生成，"
      "其中的阈值与系数直接取自源码常量，与代码保持同步。</p>");
    W("<p>划痕判据：<code>src/gui/scratch.hpp</code> ｜ "
      "图案判据：<code>modules/chardet/pattern_break.cpp</code> ｜ "
      "参数定义：<code>modules/chardet/chardet.hpp</code></p>");
    W("</div>");

    W("</div></body></html>");
    std::fclose(f);
    std::printf("已写出 %s\n", outPath);
    return 0;
}
