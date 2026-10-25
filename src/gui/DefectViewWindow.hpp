#pragma once

// 缺陷标注检查器（独立 GUI）
//
// 用途：载入精修后的模型与产品图，运行检测，把缺墨/多墨/多出的字符
//       直接标注在图上，供人工核对。
//
// 与 markloc_splitcheck 的分工：
//   splitcheck —— 建模（训练、画选区、精对齐、导出）
//   defectview（本工具）—— 用建好的模型检查产品，只读模型不改动

#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QString>
#include <QStringList>

#include <atomic>
#include <vector>

#include <opencv2/core.hpp>

#include "chardet.hpp"
#include "gui/batchcore.hpp"

class QLabel;
class QLineEdit;
class QListWidget;
class QSpinBox;
class QCheckBox;
class QScrollArea;
class QComboBox;
class QPushButton;
class QDialog;
class QProgressBar;
class QTableWidget;
class QTabWidget;
class QTextEdit;
class QStackedWidget;
class QSplitter;

/// 显示图片并可缩放的标签
class ImageView : public QLabel {
    Q_OBJECT
public:
    explicit ImageView(QWidget* parent = nullptr);
    void setImage(const QImage& img);
    void setZoom(double z);
    double zoom() const { return zoom_; }
    void fitToWindow();
    /// 缩放后图片比视口大，需要拖动才能看到别处。
    /// 左键拖会触发 QLabel 的文本选择，所以用中键/右键拖，滚轮只负责缩放。
    void panBy(int dx, int dy);
    bool imageIsEmpty() const { return src_.isNull(); }
    QSize sourceSize() const { return src_.size(); }
    bool manualZoom() const { return userZoomed_; }
    /// 还没有图（窗口刚建、还没选图片）时，在空白区给一句提示，
    /// 免得"完全空白"分不清是加载失败还是没选图
    void setPlaceholder(const QString& text);

protected:
    void wheelEvent(QWheelEvent* e) override;
    void resizeEvent(QResizeEvent* e) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;

private:
    void apply();
    /// 按滚动区视口算"整张图刚好放得下"的缩放；视口未就绪时返回 1.0
    double computeFitZoom() const;
    QImage src_;
    double zoom_ = 1.0;
    QString placeholder_;
    bool userZoomed_ = false;   // 手动缩放过就不再自动 fit
    bool panning_ = false;
    QPoint panAnchor_;          // 按下时的全局坐标
    QPoint panScrollAnchor_;    // 按下时的滚动条位置
};

class DefectViewWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit DefectViewWindow(QWidget* parent = nullptr);

    // ---- 自检入口（只给 --selftest-ui 用，正常交互不走这里）
    //
    // 存在的理由：界面"完全空白"这种故障编译器和单测都看不见，
    // 只能真的把窗口建出来再量。这些函数就是把量点暴露出来。
    bool loadImageForTest(const QString& path) { return loadImageFile(path); }
    /// 设模型目录并载入（正常交互是点「选择模型…」，这里跳过对话框）
    bool loadModelForTest(const QString& dir);
    bool modelReadyForTest() const { return modelReady_; }
    void runDetectForTest() { runDetect(); }
    /// 图像区当前显示内容的像素尺寸；未纳入布局时返回 0×0
    QSize findImageViewSize() const;
    /// 滚动区视口尺寸（图片可用的可见区域）
    QSize viewportSizeForTest() const;
    /// 状态栏当前文字
    QString statusTextForTest() const;
    /// 图像区左上角在屏幕上的位置（孤儿布局时期是 (0,0)）
    QPoint findImageViewGlobalPos() const;
    /// 把图像区当前画面抓成图（没渲染时是空图）
    QImage grabImageView() const;
    /// 最近一次生成的对比图（没跑过检测时为空图）
    QImage resultImageForTest() const;
    /// 分数面板的表格（自检用：检查行数、列数、内容是否非空）
    QTableWidget* scoreTableForTest() const { return scoreTable_; }
    /// 分数面板当前的标题文字
    QString scoreTitleForTest() const { return scoreTitle_ ? scoreTitle_->text() : QString(); }
    /// 分数面板空态文字及其可见性。
    QString scoreEmptyTextForTest() const { return scoreEmpty_ ? scoreEmpty_->text() : QString(); }
    bool scoreEmptyVisibleForTest() const { return scoreEmpty_ && !scoreEmpty_->isHidden(); }
    /// 下方容器当前是哪一页：0=分数面板 1=滑动列表
    int bottomPageForTest() const;
    /// 当前幻灯片下标
    int slideIndexForTest() const { return slideIndex_; }
    /// 幻灯片右侧图片的尺寸（未渲染时为空）
    QSize slideViewSizeForTest() const;
    /// 幻灯片当前项的缩略图（未经缩放），用于检查标签是否画出
    QImage grabSlidePixmapForTest() const;
    /// 滑块列表项数
    int batchListCountForTest() const;
    /// 以「打开大图对比」的方式打开第 row 行（自检用）
    void openRowForTest(int row) { openBatchRow(row); }
    /// 滑块列表里某一项的显示文字（"NG 确定 …" 这种），用来断言 NG/OK 标签在最前
    QString batchListTextForTest(int i) const;
    /// 缩放/拖动链路自检：滚轮放大 → 拖动 → 复位，逐步量
    bool zoomPanForTest();
    /// 连读某个控件及其祖先上的 canvas 状态；没有则返回 false
    QSize canvasSizeForTest() const;
    /// 滑动列表右侧当前显示的对比图（已缩放到显示尺寸）
    QImage slideViewImageForTest() const;

private slots:
    void chooseModelDir();
    void chooseImage();
    void runDetect();
    /// 批量检测：选一个文件夹，逐张检测并输出汇总表
    void runBatch();
    /// 取消批量检测
    void cancelBatch();
    /// 把批量结果填入表格
    void fillBatchTable();
    /// 双击/回车打开某一行的对比图
    void openBatchRow(int row);
    /// 按分类筛选批量结果表
    void applyBatchFilter();
    /// 分数面板：把上一次单张检测的划痕填进下方表格
    void fillScoreTable();
    /// 幻灯片上翻/下翻
    void slidePrev();
    void slideNext();
    void onViewChanged(int idx);
    void zoomIn();
    void zoomOut();
    void fitView();
    /// 打开灵敏度参数设置
    void openSensitivity();
    /// 把当前灵敏度保存到配置文件
    void saveSensitivity();
    /// 从配置文件载入灵敏度（不存在则用默认值）
    void loadSensitivity();

private:
    void buildUi();
    void loadModel();
    void showImage(const cv::Mat& bgr);
    void refreshList();
    void updateStatus();
protected:
    /// 拦截滚动区视口尺寸变化：视口变了才重新适配图片缩放
    bool eventFilter(QObject* obj, QEvent* ev) override;
    /// 读入一张图（供批量结果双击跳转用）
    bool loadImageFile(const QString& path);
    /// 把界面上的灵敏度换成划痕参数（数值一一对应，不做换算）
    markloc::ScratchParams scratchParams() const;

    QString modelDir_;
    QString imagePath_;

    chardet::Model model_;
    bool modelReady_ = false;
    std::vector<chardet::CharUnit> units_;

    cv::Mat gray_;        // 待检原图
    cv::Mat signal_;      // 笔画信号
    cv::Mat rawBgr_;      // 原始彩色图
    cv::Mat resultBgr_;   // 带标注的图（当前显示的内容）
    QString comparePath_; // 对比图保存路径

    struct Hit {
        QString kind;     // 缺墨/多墨/多出字符
        QString detail;
        QString where;
        double score = 0.0;
    };
    std::vector<Hit> hits_;

    /// 各检测项的灵敏度参数
    ///
    /// 这些阈值是"判多严"的关口：调高 = 更宽松（漏检增多、误报减少），
    /// 调低 = 更严格（检出增多、误报也增多）。
    /// 实测这批数据的 OK 件自带刻印抖动与轻微划痕，
    /// 具体切点需用客户的真实 NG 样本标定。
    struct Sensitivity {
        // 逐字符精对齐（解决"框没对准字形"）
        bool perCharAlign = true;
        // 多模板双指标（检缺笔画 / 改笔画）
        // 原理：拿待检字符与模板库比，每字符得到最小差异 d_c；
        //       maxDiff = 最大 d_c（最可疑的字），secondDiff = 第二大 d_c。
        //       「某字刻错」→ max 远大于 second；「整体波动」→ 两者接近。
        //
        // 【默认关闭判 NG】2026-10-21 全量实测结论：
        //   30 张 OK 的比值分布 0.85~1.70，8 张真实 NG(划痕型) 仅 1.0~1.39，
        //   两者完全重叠。原定门槛 1.5 会让 OK 误报 6.7%，且真实 NG 零检出。
        //   早期看到的 25x / 2.5x 判别力来自合成样本 SYN0012_*
        //   （由单张 OK 改一个字生成，其余 21 字与模板完全一致 → secondDiff 被压到 0.03），
        //   属于数据泄漏，不能代表真实分布。
        //   因此本判据暂不参与 NG 判定，只保留"可疑位置"提示，供人工复核。
        bool checkMultiTmpl = true;   // 是否运行多模板比对（仅用于标注）
        bool multiDecideNg = false;   // 是否允许它判 NG（默认否）
        double multiRatio = 1.5;      // maxDiff / secondDiff 门槛
        double multiWarn = 0.15;      // 单字符 d_c 超过此值标为"可疑"
        //   标定依据：OK 的 maxDiff 实测 0.0991~0.2003，
        //   取 0.15 可覆盖约八成 OK 的正常波动，再高就无提示价值。

        // 整字数量
        bool checkRowCount = true;

        // 划痕
        bool checkScratch = true;
        double scratchMinLen = 60.0;      // 最短划痕长度(px)
        double scratchMinCov = 0.25;      // 最短覆盖率
        // 划痕跨字符重连核大小(px)。
        //
        // 减去模型笔画会把一条跨字划痕切成碎片（压在字上的部分全被当笔画删掉），
        // 碎片各自都过不了长度门槛，整条划痕就消失了。闭运算沿划痕方向把断口
        // 接回去，让同一条划痕重新连成整体。核要大于笔画宽度才能跨过一根竖笔。
        // 0 = 关闭重连（退回旧行为，跨字划痕会漏检）。
        int scratchBridge = 25;
        // 能量门槛。
        //
        // 已退成纯去噪门槛，默认 0（不筛）。原来按 84 张 OK 标定到 30 万，
        // 实测让 8 张真 NG 全部漏检：真划痕能量只有 1.7 万~24.4 万，
        // 标定集里根本没出现过。严重程度改由下面的"深度 × 跨字数"判定，
        // 这里不再承担判别职责，留着只为在脏背景下能先筛掉零星噪点。
        double scratchMinEnergy = 0.0;
        // 划痕灰度门槛：低于此值视为浅表擦痕，放行
        double scratchMinGray = 0.45;

        // 划痕严重度报告门槛 = 深度 × 跨越字数。
        //
        // 单项都不成立：只看深度，OK 件里离字很远的深擦痕会排到最前（实测
        // OK 最深 1204，NG 只有 78~1038）；只看跨字数，则一条浅线也能过关。
        //
        // 实测全量（84 张 OK + 7 张 NG，均为 v00）：
        //   NG     2289 2256 2209 2040 1321 1219        ← 0110 只有 73
        //   OK     1430 1182 1080 873 856 847 ...       ← 前三张是 0109_RT/
        //                                                  0109_FT/0101_FT
        //   —— 注意 0112_RT(1219) 比 0109_RT(1430) 还低，两者是真的重叠，
        //      不存在能把它们分开的门槛。0109 那两道痕的深度确实到了
        //      477/591、跨度 3 个字，与真划痕同量级，这属于"痕迹本身就像"，
        //      不是阈值没调好。
        //
        // 门槛取 1000：真 NG 6/6 全报（0110 客户已确认可以漏），
        //   84 张 OK 里多报 3 张 —— 0101 客户判定确实是缺陷（属正确检出），
        //   0109 的两次拍摄是真误报，误报率 2/84 ≈ 2.4%。
        // 想彻底不误报就把门槛抬到 1500：0109 消失，但 0112 两张跟着漏，
        //   真 NG 掉到 4/6。这个取舍按"宁可误报不能漏检"取 1000。
        double scratchMinComposite = 1000.0;
        // 四档切点（严重度 = 深度 × 跨越字数）。
        //
        // 做成可调，是因为判 NG 的线放在哪里属于客户的决定，不该由我在代码里
        // 钉死。界面上每个缺陷旁边会直接写出「要让这一条升级/降级，切点需要
        // 调到多少」，调完即可对照，不必反复猜。
        //
        // 切点铺在实际会报告的那段区间上，让"深浅"看得出来：
        // 实测 1219/1321/1430 → 可疑、明显；2040 以上 → 确定。
        double scratchCutCertain = 1900.0;
        double scratchCutObvious = 1300.0;
        double scratchCutSuspect = 1000.0;
        // 深度测量：垂直剖面的半长(px)。要大于划痕宽度、小于字宽。
        int scratchDepthReach = 12;

        // 图案区断点（logo 椭圆环）
        //
        // 判的是"这条闭合曲线被切断没有"，与划痕是两套独立判据：
        // 分数不同量纲，各自过各自的门槛，任一达标就判 NG。
        bool checkPattern = true;
        chardet::PatternBreakParams pattern;

        // 污损
        bool checkDirt = true;
        int dirtMinArea = 120;            // 最小污损面积(px)

        // 溢料
        bool checkAppendage = true;
        int appendMinArea = 80;
        int appendMaxArea = 3000;
        double appendMinElong = 2.2;      // 最小长宽比

        // 局部缺墨（大框专用，如 logo）
        //
        // 按 30×30 小格逐块比较，解决"大框稀释局部缺口"的问题。
        // 整框比例判据对 logo 无效：15×10 的缺口只占 1~2%。
        bool checkLocalMiss = true;
        double localMissRatio = 0.40;     // 格内墨量低于模型的此比例即判缺墨

        // 字符级（缺墨/多墨）
        // 逐字符对齐修好后，此项的误报已有改善；具体水平见使用说明的回归数据
        bool checkCharLevel = true;
    };
    Sensitivity sens_;
    QPushButton* sensBtn_ = nullptr;
    QPushButton* batchBtn_ = nullptr;
    QPushButton* cancelBtn_ = nullptr;
    QProgressBar* progressBar_ = nullptr;

    QTabWidget* tabs_ = nullptr;
    QTableWidget* batchTable_ = nullptr;   // 批量结果表（明细视图的备用）
    QTextEdit* statsView_ = nullptr;       // 参数与性能
    QComboBox* batchFilter_ = nullptr;     // 批量结果分类筛选
    QLabel* batchSummary_ = nullptr;       // 批量结果分类计数

    // ---- 下方：缺陷分数面板（单张检测结果）
    //
    // 图片是 3.4:1 的宽图，在接近方形的视口里纵向天然留白。
    // 那块空白正好放"这一张报出了什么、分数多少、切点调到多少会变档"，
    // 比让人在图上凑近看小字强。
    QTableWidget* scoreTable_ = nullptr;
    QLabel* scoreTitle_ = nullptr;
    /// 分数面板的空态文字。没有划痕时表格收起、只显示它，
    /// 避免"8 列表头 + 一片空白行"被误读成数据没加载出来。
    QLabel* scoreEmpty_ = nullptr;
    /// 批量页左侧列表下方的批次汇总（NG/OK 计数 + 档位分布）。
    ///
    /// 与筛选行里的 batchSummary_ 不是一回事：那个是"当前筛选后显示几张"，
    /// 这个整批的结论。列表只有三五项时下方会空掉一大块，用它补上。
    QLabel* batchFoot_ = nullptr;
    /// 上一次单张检测的划痕明细（原始结构，供分数面板与升降档计算）
    std::vector<markloc::ScratchHit> scratchHits_;
    /// 上一次单张检测的图案区明细（与划痕并列的第二套判据）
    std::vector<chardet::PatternBreakResult> patternResults_;
    /// 上一轮整图配准的位移（相位相关原始值）。
    /// 图案区要按 -lastAlign 平移，符号见 runDetect() 里的说明。
    double lastAlignX_ = 0.0;
    double lastAlignY_ = 0.0;

    // ---- 下方：批量结果的滑动列表
    //
    // 批量页原来只有表格，逐项翻看得靠滚动条来回拖。补一个左右可切的
    // 幻灯片式浏览：一次看一项、上下翻或点按钮都行，双击直接进单张对比。
    QListWidget* batchList_ = nullptr;     // 左侧缩略列表（可滚动）
    // 右侧当前项的对比图。原来是个 QLabel 塞 pixmap，缩放时只能重算 pixmap、
    // 无处存缩放状态，所以"点开看不大、看不了别处"。换成可缩放/可拖动的视图，
    // 和单张页的 ImageView 同一套语义（滚轮缩放、按住中键拖、复位）。
    QScrollArea* slideScroll_ = nullptr;
    ImageView* slideView_ = nullptr;
    QLabel* slideInfo_ = nullptr;          // 右侧当前项的文字说明
    QStackedWidget* bottomStack_ = nullptr;// 下方容器：分数面板 / 批量列表 切换
    /// 上下分割条。批量结束后要把下方撑大，否则滑动列表的对比图
    /// 会被压在 130px 高度的缝里，放大也无从谈起。
    QSplitter* leftSplit_ = nullptr;
    int slideIndex_ = -1;                  // 当前幻灯片在 batchRows_ 里的下标
    /// 批量结束后把结果灌进滑动列表（table 那套之外的另一路展示）
    void fillBatchList();
    /// 显示第 i 项（i 越界则夹到有效范围）
    void showSlide(int i);
    /// 生成某一项的对比图（复用单张检测的对比图逻辑）
    QPixmap makeSlidePixmap(int row);
    /// 最近一次生成的幻灯片原图（未缩放），自检用
    QImage currentSlideImage_;

    /// 批量检测的状态
    std::atomic<bool> batchCancel_{false};
    bool batching_ = false;

    /// 上一次批量检测的汇总（供表格与统计展示）
    ///
    /// 结构定义在 gui/batchcore.hpp，因为 --batch 命令行模式要用同一份。
    using BatchRow = markloc::BatchRow;
    std::vector<BatchRow> batchRows_;
    double batchTotalMs_ = 0.0;
public:
    /// 直接灌入一批批量结果，用于在没有真实批量跑动时验证滑动列表
    void injectBatchRowsForTest(const std::vector<markloc::BatchRow>& rows);
private:
    /// 上一次批量检测选的文件夹（CSV 与跳转都用它）
    QString batchDir_;
    /// 批量结果里出现过的分类（与 batchFilter_ 的选项一一对应）
    QStringList batchClasses_;

    /// 多模板双指标的最近结果（供统计页显示）
    double multiMax_ = 0.0;
    double multiSecond_ = 0.0;
    double multiRatio_ = 0.0;

    QLineEdit* modelEdit_ = nullptr;
    QLineEdit* imageEdit_ = nullptr;
    QComboBox* viewCombo_ = nullptr;
    QScrollArea* scroll_ = nullptr;
    ImageView* imgView_ = nullptr;
    QListWidget* list_ = nullptr;
    QLabel* status_ = nullptr;
    QCheckBox* showOkChk_ = nullptr;
    QCheckBox* charLevelChk_ = nullptr;   // 是否启用字符级判据（默认关）
    QSpinBox* zoomSpin_ = nullptr;
    int resultView_ = 2;   // 0=模型 1=原图 2=标注
};
