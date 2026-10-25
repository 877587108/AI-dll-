cmake_minimum_required(VERSION 3.20)
project(markloc VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# All source files are UTF-8 (including Chinese diagnostics and model labels).
# Keep this at directory scope so diagnostic targets added later in this file
# receive the same compiler contract as the core libraries.
if (MSVC)
    add_compile_options(/W4 /permissive- /utf-8 /Zc:__cplusplus)
endif()

# ============================================================ dependencies
set(OpenCV_ROOT "D:/opencv" CACHE PATH "OpenCV install root")
if (NOT OpenCV_DIR)
    set(OpenCV_DIR "${OpenCV_ROOT}/build/x64/vc16/lib")
endif()
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version : ${OpenCV_VERSION}")

# ============================================================ modular core libraries
#
# Dependency direction:
#   markloc_chardet -> markloc_engine -> markloc_application -> markloc_core(compat)
#
# `markloc_core` remains an INTERFACE compatibility target, so existing tools keep
# their old link line. New code should link `markloc_application` and include
# `core/api.hpp`. A new algorithm can replace the application adapter without
# recompiling the image/model implementation in markloc_engine.
set(MARKLOC_CHARDET_SOURCES
    modules/chardet/split.cpp
    modules/chardet/match.cpp
    modules/chardet/blocks.cpp
    modules/chardet/skeleton.cpp
    modules/chardet/pattern_break.cpp
    modules/chardet/multitmpl.cpp
    modules/chardet/align_char.cpp
    modules/chardet/model.cpp
    modules/chardet/pattern/pattern_types.cpp
    modules/chardet/pattern/preprocessor.cpp
    modules/chardet/pattern/profile_builder.cpp
    modules/chardet/pattern/pattern_pipeline.cpp
    modules/chardet/pattern/det_notch_window.cpp
    modules/chardet/pattern/det_gap_run.cpp
    modules/chardet/pattern/det_registry.cpp
)
add_library(markloc_chardet STATIC ${MARKLOC_CHARDET_SOURCES})
target_include_directories(markloc_chardet PUBLIC modules/chardet ${OpenCV_INCLUDE_DIRS})
target_link_libraries(markloc_chardet PUBLIC ${OpenCV_LIBS})

add_library(markloc_engine STATIC
    src/core/signal.cpp
    src/core/model_io.cpp
    src/core/analyze.cpp
    src/core/explain.cpp
    src/core/build.cpp
    src/core/char_detector.cpp
    src/core/warp.cpp
    src/core/detector_suite.cpp
    src/core/detect_appendage.cpp
    src/core/detect_scratch.cpp
    src/core/detect_missing.cpp
)
target_include_directories(markloc_engine PUBLIC src modules/chardet ${OpenCV_INCLUDE_DIRS})
target_link_libraries(markloc_engine PUBLIC markloc_chardet ${OpenCV_LIBS})

add_library(markloc_application STATIC
    src/core/algorithm_types.cpp
    src/core/default_detection_algorithm.cpp
    src/core/naming.cpp
    src/core/model_repository.cpp
    src/core/pipeline.cpp
)
target_include_directories(markloc_application PUBLIC src modules/chardet ${OpenCV_INCLUDE_DIRS})
target_link_libraries(markloc_application PUBLIC markloc_engine)

# Compatibility umbrella for existing command-line diagnostics and legacy GUI.
add_library(markloc_core INTERFACE)
target_include_directories(markloc_core INTERFACE src modules/chardet ${OpenCV_INCLUDE_DIRS})
target_link_libraries(markloc_core INTERFACE markloc_application)

add_library(markloc::chardet ALIAS markloc_chardet)
add_library(markloc::engine ALIAS markloc_engine)
add_library(markloc::application ALIAS markloc_application)

set(ALL_TARGETS markloc_chardet markloc_engine markloc_application)

# ============================================================ public contract tests
include(CTest)
if (BUILD_TESTING)
    add_executable(markloc_public_contract_test tests/public_contract_test.cpp)
    target_link_libraries(markloc_public_contract_test PRIVATE markloc_application)
    target_include_directories(markloc_public_contract_test PRIVATE src)
    add_test(NAME markloc.public_contract COMMAND markloc_public_contract_test)
    list(APPEND ALL_TARGETS markloc_public_contract_test)
endif()

# ============================================================ command line tools
# Tuning and diagnostic helpers. Each links only the core library.
set(TOOL_NAMES eval mkmodel detect render flow flow2 flow3 charalign_test)
foreach (tool ${TOOL_NAMES})
    add_executable(markloc_${tool} tools/${tool}.cpp)
    target_link_libraries(markloc_${tool} PRIVATE markloc_core)
    list(APPEND ALL_TARGETS markloc_${tool})
endforeach()

# ============================================================ Qt GUI
find_package(Qt6 QUIET COMPONENTS Widgets Gui Core)

if (Qt6_FOUND)
    # AUTOMOC is disabled on purpose: in restricted sandboxes cmake_autogen
    # cannot spawn subprocesses, so moc is invoked explicitly below.

    # ---- evaluation app: recognition + performance statistics
    add_executable(markloc_gui
        src/gui/main.cpp
        src/gui/TrainWindow.cpp
        src/gui/TrainWindow.hpp
    )
    target_link_libraries(markloc_gui PRIVATE markloc_core Qt6::Widgets Qt6::Gui Qt6::Core)
    target_include_directories(markloc_gui PRIVATE src)

    # ---- model build app: training / region selection / char splitting
    add_executable(markloc_train
        src/gui/train_main.cpp
        src/gui/TrainToolWindow.cpp
        src/gui/TrainToolWindow.hpp
        src/gui/CharBoxEditor.cpp
        src/gui/CharBoxEditor.hpp
        src/gui/BoxCanvas.cpp
        src/gui/BoxCanvas.hpp
    )
    target_link_libraries(markloc_train PRIVATE markloc_core Qt6::Widgets Qt6::Gui Qt6::Core)
    target_include_directories(markloc_train PRIVATE src)

    # ---- region split verification app (customer facing)
    add_executable(markloc_splitcheck
        src/gui/splitverify_main.cpp
        src/gui/SplitVerifyWindow.cpp
        src/gui/SplitVerifyWindow.hpp
        src/gui/BoxCanvas.cpp
        src/gui/BoxCanvas.hpp
    )
    target_link_libraries(markloc_splitcheck PRIVATE markloc_core Qt6::Widgets Qt6::Gui Qt6::Core)
    target_include_directories(markloc_splitcheck PRIVATE src)

    # ---- defect annotation viewer (read-only, consumes a trained model)
    add_executable(markloc_defectview
        src/gui/defectview_main.cpp
        src/gui/DefectViewWindow.cpp
        src/gui/DefectViewWindow.hpp
        src/gui/batchcore.hpp
        src/gui/scratch.hpp
    )
    target_link_libraries(markloc_defectview PRIVATE markloc_core Qt6::Widgets Qt6::Gui Qt6::Core)
    target_include_directories(markloc_defectview PRIVATE src)

    # ---- locate plugin debugger (locate plugin + model + image visual compare)
    add_executable(markloc_locateview
        src/gui/locateview_main.cpp
        src/gui/LocateViewWindow.cpp
        src/gui/LocateViewWindow.hpp
    )
    target_link_libraries(markloc_locateview PRIVATE markloc_core Qt6::Widgets Qt6::Gui Qt6::Core)
    target_include_directories(markloc_locateview PRIVATE src include)

    # ---- flow document generator (QPainter, so labels can be Chinese)
    add_executable(markloc_flowdoc tools/flowdoc.cpp)
    target_link_libraries(markloc_flowdoc PRIVATE markloc_core Qt6::Widgets Qt6::Gui Qt6::Core)
    target_include_directories(markloc_flowdoc PRIVATE src modules/chardet tools)

    add_executable(markloc_flowprinciple tools/flowprinciple.cpp)
    target_link_libraries(markloc_flowprinciple PRIVATE markloc_core Qt6::Widgets Qt6::Gui Qt6::Core)
    target_include_directories(markloc_flowprinciple PRIVATE src modules/chardet tools)

    add_executable(markloc_p0ana tools/p0ana.cpp)
    target_link_libraries(markloc_p0ana PRIVATE markloc_core)
    target_include_directories(markloc_p0ana PRIVATE src modules/chardet tools)

    add_executable(markloc_p0char tools/p0char.cpp)
    target_link_libraries(markloc_p0char PRIVATE markloc_core)
    target_include_directories(markloc_p0char PRIVATE src modules/chardet tools)

    add_executable(markloc_multitmpl tools/multitmpl.cpp)
    target_link_libraries(markloc_multitmpl PRIVATE markloc_core)
    target_include_directories(markloc_multitmpl PRIVATE src modules/chardet tools)

    add_executable(markloc_dual tools/dual.cpp)
    target_link_libraries(markloc_dual PRIVATE markloc_core)
    target_include_directories(markloc_dual PRIVATE src modules/chardet tools)

    add_executable(markloc_mkmultitmpl tools/mkmultitmpl.cpp)
    target_link_libraries(markloc_mkmultitmpl PRIVATE markloc_core)
    target_include_directories(markloc_mkmultitmpl PRIVATE src modules/chardet tools)

    add_executable(markloc_chardet_eval tools/chardet_eval.cpp)
    target_link_libraries(markloc_chardet_eval PRIVATE markloc_core)
    target_include_directories(markloc_chardet_eval PRIVATE src modules/chardet tools)

    add_executable(markloc_diagscratch tools/diagscratch.cpp)
    target_link_libraries(markloc_diagscratch PRIVATE markloc_core)
    target_include_directories(markloc_diagscratch PRIVATE src modules/chardet tools)

    add_executable(markloc_scratchcmp tools/scratchcmp.cpp)
    target_link_libraries(markloc_scratchcmp PRIVATE markloc_core)
    target_include_directories(markloc_scratchcmp PRIVATE src modules/chardet tools)

    add_executable(markloc_suspect tools/suspect.cpp)
    target_link_libraries(markloc_suspect PRIVATE markloc_core)
    target_include_directories(markloc_suspect PRIVATE src modules/chardet tools)

    add_executable(markloc_grilleval tools/grilleval.cpp)
    target_link_libraries(markloc_grilleval PRIVATE markloc_core)
    target_include_directories(markloc_grilleval PRIVATE src modules/chardet tools)

    add_executable(markloc_grillediag tools/grillediag.cpp)
    target_link_libraries(markloc_grillediag PRIVATE markloc_core)
    target_include_directories(markloc_grillediag PRIVATE src modules/chardet tools)

    add_executable(markloc_annotedemo tools/annotedemo.cpp)
    target_link_libraries(markloc_annotedemo PRIVATE markloc_core)
    target_include_directories(markloc_annotedemo PRIVATE src modules/chardet tools)

    # Discrimination region plot: severity distribution of OK/NG,
    add_executable(markloc_bandplot tools/bandplot.cpp)
    target_link_libraries(markloc_bandplot PRIVATE ${OpenCV_LIBS})
    target_include_directories(markloc_bandplot PRIVATE src modules/chardet tools)

    # Char-box fit diagnostic: does the box actually contain the glyph.
    # Judgement logic: if the ink-region box is off, every criterion is just noise, because the box is measured against the wrong place.
    # Need a tool that reports the box offsets numerically, not by eyeballing.
    add_executable(markloc_boxdiag tools/boxdiag.cpp)
    target_link_libraries(markloc_boxdiag PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_boxdiag PRIVATE src modules/chardet tools)

    # Cross-check the two detection paths: interactive 10 hits vs batch 1.
    add_executable(markloc_scratchcmp2 tools/scratchcmp2.cpp)
    target_link_libraries(markloc_scratchcmp2 PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_scratchcmp2 PRIVATE src modules/chardet tools)

    # False-positive trace: 0109 lives in ok/ but was judged NG.
    #   scratchtrace  閫愭潯瑁佸浘 + 淇″彿/妯″瀷棰戠巼缁熻
    #   freqvs        signal vs model-frequency side by side
    #   maskeval      mask coverage for one file (parameter sweep)
    #   masksweep     mask coverage across the whole set
    # Keeping these tools for reproducibility.
    add_executable(markloc_scratchtrace tools/scratchtrace.cpp)
    target_link_libraries(markloc_scratchtrace PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_scratchtrace PRIVATE src modules/chardet tools)

    add_executable(markloc_freqvs tools/freqvs.cpp)
    target_link_libraries(markloc_freqvs PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_freqvs PRIVATE src modules/chardet tools)

    add_executable(markloc_maskeval tools/maskeval.cpp)
    target_link_libraries(markloc_maskeval PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_maskeval PRIVATE src modules/chardet tools)

    add_executable(markloc_masksweep tools/masksweep.cpp)
    target_link_libraries(markloc_masksweep PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_masksweep PRIVATE src modules/chardet tools)

    # Percentile survey of the stroke signal inside the 22 char boxes.
    # Purpose: pick the upper bound of an adaptive strokeHi = p90 from measured data,
    # not from a guess.
    add_executable(markloc_p90stat tools/p90stat.cpp)
    target_link_libraries(markloc_p90stat PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_p90stat PRIVATE src modules/chardet tools)

    # Lower/upper bound derivation from measured data, not from guesswork.
    add_executable(markloc_boundmgr tools/boundmgr.cpp)
    target_link_libraries(markloc_boundmgr PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_boundmgr PRIVATE src modules/chardet tools)

    # Normalisation-scale derivation: which formula makes strokeHi meaningful.
    add_executable(markloc_scalenorm tools/scalenorm.cpp)
    target_link_libraries(markloc_scalenorm PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_scalenorm PRIVATE src modules/chardet tools)

    # Signal vs model-frequency overlay: decide whether the leak is a glyph or a scratch.
    add_executable(markloc_sigoverlay tools/sigoverlay.cpp)
    target_link_libraries(markloc_sigoverlay PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_sigoverlay PRIVATE src modules/chardet tools)

    # Bounding-box vs true-line character-crossing test.
    add_executable(markloc_boxleak tools/boxleak.cpp)
    target_link_libraries(markloc_boxleak PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_boxleak PRIVATE src modules/chardet tools)

    # Logo-region stability: is a 279x177 non-glyph box even maskable?
    add_executable(markloc_logostab tools/logostab.cpp)
    target_link_libraries(markloc_logostab PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_logostab PRIVATE src modules/chardet tools)

    # Per-char alignment probe: does aligning boxes leave the pixels misaligned?
    add_executable(markloc_alignprobe tools/alignprobe.cpp)
    target_link_libraries(markloc_alignprobe PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_alignprobe PRIVATE src modules/chardet tools)

    # Single-image fast iteration loop: stage-by-stage counts for one file.
    add_executable(markloc_quick tools/quick.cpp)
    target_link_libraries(markloc_quick PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_quick PRIVATE src modules/chardet tools)

    # ref vs freq as the phase-correlation reference: which predicts real glyphs?
    add_executable(markloc_refcmp tools/refcmp.cpp)
    target_link_libraries(markloc_refcmp PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_refcmp PRIVATE src modules/chardet tools)

    # How much real ink the model mask actually covers, per region.
    add_executable(markloc_freqcov tools/freqcov.cpp)
    target_link_libraries(markloc_freqcov PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_freqcov PRIVATE src modules/chardet tools)

    # Where the model-mask leak actually sits, and which blobs survive bridging.
    add_executable(markloc_leakwhere tools/leakwhere.cpp)
    target_link_libraries(markloc_leakwhere PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_leakwhere PRIVATE src modules/chardet tools)

    # Same pipeline, different signal normalisation: does the scale explain the leak?
    add_executable(markloc_normtest tools/normtest.cpp)
    target_link_libraries(markloc_normtest PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_normtest PRIVATE src modules/chardet tools)

    # Per-hit coordinates replicating detectOne exactly, incl. per-char box moves.
    add_executable(markloc_hitwhere tools/hitwhere.cpp)
    target_link_libraries(markloc_hitwhere PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_hitwhere PRIVATE src modules/chardet tools)

    # charsHit rule comparison: bbox-intersection vs true segment crossing.
    add_executable(markloc_crossrule tools/crossrule.cpp)
    target_link_libraries(markloc_crossrule PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_crossrule PRIVATE src modules/chardet tools)

    # Glyph-edge leak: weak-threshold vs model-binarisation threshold mismatch.
    add_executable(markloc_edgeleak tools/edgeleak.cpp)
    target_link_libraries(markloc_edgeleak PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_edgeleak PRIVATE src modules/chardet tools)

    # Why the circle mark is absent from the model: per-image occurrence rate.
    add_executable(markloc_dotstat tools/dotstat.cpp)
    target_link_libraries(markloc_dotstat PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_dotstat PRIVATE src modules/chardet tools)

    # Paint real hits onto the raw image so the cause can be seen, not inferred.
    add_executable(markloc_hitpaint tools/hitpaint.cpp)
    target_link_libraries(markloc_hitpaint PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_hitpaint PRIVATE src modules/chardet tools)

    # Nearest-neighbour crop zoom for inspecting small regions.
    add_executable(markloc_cropzoom tools/cropzoom.cpp)
    target_link_libraries(markloc_cropzoom PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_cropzoom PRIVATE src modules/chardet tools)

    # Per-stage visualisation of the scratch pipeline inside one crop window.
    add_executable(markloc_chain tools/chain.cpp)
    target_link_libraries(markloc_chain PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_chain PRIVATE src modules/chardet tools)

    # Threshold mismatch: how wide is the leaked glyph-edge band?
    add_executable(markloc_thrmatch tools/thrmatch.cpp)
    target_link_libraries(markloc_thrmatch PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_thrmatch PRIVATE src modules/chardet tools)

    # Controlled fix experiment: does resampling the signal kill the false hit?
    add_executable(markloc_fixexp tools/fixexp.cpp)
    target_link_libraries(markloc_fixexp PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_fixexp PRIVATE src modules/chardet tools)

    # Trace a hit line back through every stage to find what it is actually made of.
    add_executable(markloc_trace tools/trace.cpp)
    target_link_libraries(markloc_trace PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_trace PRIVATE src modules/chardet tools)

    # Why does a real stroke have freq=0? Per-image threshold hit-rate along it.
    add_executable(markloc_strokefreq tools/strokefreq.cpp)
    target_link_libraries(markloc_strokefreq PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_strokefreq PRIVATE src modules/chardet tools)

    # Mask-threshold sweep: which freq cutoff stops real strokes leaking?
    add_executable(markloc_maskeval2 tools/maskeval2.cpp)
    target_link_libraries(markloc_maskeval2 PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_maskeval2 PRIVATE src modules/chardet tools)

    # Discriminating metric: leak as a ratio, plus stroke-leak (bright but freq<thr).
    add_executable(markloc_leakratio tools/leakratio.cpp)
    target_link_libraries(markloc_leakratio PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_leakratio PRIVATE src modules/chardet tools)

    # Per-hit decomposition for 0109_RT vs 0109_FT: are the hits one class or two?
    add_executable(markloc_split0109 tools/split_0109.cpp)
    target_link_libraries(markloc_split0109 PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_split0109 PRIVATE src modules/chardet tools)

    # Does the NUMBER of passed hits separate clean parts from真 defects?
    add_executable(markloc_hitcount tools/hitcount.cpp)
    target_link_libraries(markloc_hitcount PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_hitcount PRIVATE src modules/chardet tools)

    # Paint hits on the raw image so "is this a scratch?" becomes a visual check.
    add_executable(markloc_painthits tools/paint_hits.cpp)
    target_link_libraries(markloc_painthits PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_painthits PRIVATE src modules/chardet tools)

    # Is inkDistP50 stable across the whole library, or only on the 7 samples?
    add_executable(markloc_inkdist tools/inkdist.cpp)
    target_link_libraries(markloc_inkdist PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_inkdist PRIVATE src modules/chardet tools)

    # Per-candidate "measured vs threshold" so a gate that does not bite is visible.
    add_executable(markloc_gatewhy tools/gatewhy.cpp)
    target_link_libraries(markloc_gatewhy PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_gatewhy PRIVATE src modules/chardet tools)

    # Char-box audit: are the boxes split wrong (merged fragments / slivers / overlaps)?
    add_executable(markloc_boxaudit tools/boxaudit.cpp)
    target_link_libraries(markloc_boxaudit PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_boxaudit PRIVATE src modules/chardet tools)

    # One sweep over all minCharCover values instead of guess-rebuild-run.
    add_executable(markloc_sweepcover tools/sweep_cover.cpp)
    target_link_libraries(markloc_sweepcover PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_sweepcover PRIVATE src modules/chardet tools)

    # Is the line real matter, or leak bridged by CLOSE?
    add_executable(markloc_lineform tools/lineform.cpp)
    target_link_libraries(markloc_lineform PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_lineform PRIVATE src modules/chardet tools)

    # Direction agreement: does the line run along the glyph stroke, or across it?
    add_executable(markloc_aligndir tools/aligndir.cpp)
    target_link_libraries(markloc_aligndir PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_aligndir PRIVATE src modules/chardet tools)

    # Alignment health check: is this image actually registered before we measure anything?
    add_executable(markloc_aligncheck tools/aligncheck.cpp)
    target_link_libraries(markloc_aligncheck PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_aligncheck PRIVATE src modules/chardet tools)

    # Side-by-side template vs image comparison panel.
    add_executable(markloc_vsmodel tools/vs_model.cpp)
    target_link_libraries(markloc_vsmodel PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_vsmodel PRIVATE src modules/chardet tools)

    # Per-char match health: template size, search radius, peak, shift.
    add_executable(markloc_boxmatch tools/boxmatch.cpp)
    target_link_libraries(markloc_boxmatch PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_boxmatch PRIVATE src modules/chardet tools)

    # Which warp direction/sign is correct for per-char pixel realignment?
    add_executable(markloc_warpsign tools/warpsign.cpp)
    target_link_libraries(markloc_warpsign PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_warpsign PRIVATE src modules/chardet tools)

    # 0101 deep dive: full production alignment chain + per-hit decomposition.
    add_executable(markloc_chase0101 tools/chase0101.cpp)
    target_link_libraries(markloc_chase0101 PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_chase0101 PRIVATE src modules/chardet tools)

    # Locate regions 1/2/logo on new-spec images (1496x1496) with per-region scale+shift.
    add_executable(markloc_regionalign tools/region_align.cpp)
    target_link_libraries(markloc_regionalign PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_regionalign PRIVATE src modules/chardet tools)

    # Logo ring breakage: connected-component analysis of the ellipse.
    add_executable(markloc_logobreak tools/logo_break.cpp)
    target_link_libraries(markloc_logobreak PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_logobreak PRIVATE src modules/chardet tools)

    # Logo ring continuity separation: NG (broken ring) vs OK.
    add_executable(markloc_logosep tools/logo_sep.cpp)
    target_link_libraries(markloc_logosep PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_logosep PRIVATE src modules/chardet tools)

    # Logo breakage by per-pixel local comparison in the model's coordinate frame.
    add_executable(markloc_logobreak2 tools/logo_break2.cpp)
    target_link_libraries(markloc_logobreak2 PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_logobreak2 PRIVATE src modules/chardet tools)

    # Raw pixel diff between two images, no normalization.
    add_executable(markloc_imgdiff tools/imgdiff.cpp)
    target_link_libraries(markloc_imgdiff PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_imgdiff PRIVATE src modules/chardet tools)

    # Zoomed side-by-side crop comparison around a coordinate.
    add_executable(markloc_cropcmp tools/cropcmp.cpp)
    target_link_libraries(markloc_cropcmp PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_cropcmp PRIVATE src modules/chardet tools)

    # Position-independent break detection: morphological close reveals the gap.
    add_executable(markloc_breakscan tools/break_scan.cpp)
    target_link_libraries(markloc_breakscan PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_breakscan PRIVATE src modules/chardet tools)

    # NG vs OK separation, several position-independent metrics side by side.
    add_executable(markloc_oksep tools/ok_sep.cpp)
    target_link_libraries(markloc_oksep PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_oksep PRIVATE src modules/chardet tools)

    # Locate the plastic package body as a rigid anchor for cropping.
    add_executable(markloc_bodyloc tools/body_loc.cpp)
    target_link_libraries(markloc_bodyloc PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_bodyloc PRIVATE src modules/chardet tools)

    # Locate the plastic package body from its long straight edges.
    add_executable(markloc_bodyline tools/body_line.cpp)
    target_link_libraries(markloc_bodyline PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_bodyline PRIVATE src modules/chardet tools)

    # Locate the package body edges by horizontal/vertical projection.
    add_executable(markloc_bodyproj tools/body_proj.cpp)
    target_link_libraries(markloc_bodyproj PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_bodyproj PRIVATE src modules/chardet tools)

    # Match projection profiles of new-spec images against the existing library.
    add_executable(markloc_projmatch tools/proj_match.cpp)
    target_link_libraries(markloc_projmatch PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_projmatch PRIVATE src modules/chardet tools)

    # Crop new-spec images to model size using the logo as anchor.
    add_executable(markloc_cropfit tools/crop_fit.cpp)
    target_link_libraries(markloc_cropfit PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_cropfit PRIVATE src modules/chardet tools)

    # Command-line retraining so strokeHi becomes an experiment, not a rebuild.
    add_executable(markloc_retrain tools/retrain.cpp)
    target_link_libraries(markloc_retrain PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_retrain PRIVATE src modules/chardet tools)


    # Alignment limit probe: shift glyphs, measure how far alignment holds.
    add_executable(markloc_alignlimit tools/alignlimit.cpp)
    target_link_libraries(markloc_alignlimit PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_alignlimit PRIVATE src modules/chardet tools)

    # Vertical projection feasibility: is the profile sharp enough.
    add_executable(markloc_projcheck tools/projcheck.cpp)
    target_link_libraries(markloc_projcheck PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_projcheck PRIVATE src modules/chardet tools)

    # Per-char alignment by vertical projection of the char box (user-requested path).
    add_executable(markloc_logoshape tools/logoshape.cpp)
target_link_libraries(markloc_logoshape PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_logoshape PRIVATE src src/gui modules/chardet tools)
add_executable(markloc_fitellipse tools/fitellipse.cpp)
target_link_libraries(markloc_fitellipse PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_fitellipse PRIVATE src src/gui modules/chardet tools)
add_executable(markloc_breakprobe tools/breakprobe.cpp)
target_link_libraries(markloc_breakprobe PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_breakprobe PRIVATE src src/gui modules/chardet tools)
add_executable(markloc_fitnew tools/fitnew.cpp)
target_link_libraries(markloc_fitnew PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_fitnew PRIVATE src src/gui modules/chardet tools)
add_executable(markloc_regionfmt tools/regionfmt.cpp)
target_link_libraries(markloc_regionfmt PRIVATE markloc_core ${OpenCV_LIBS})
target_include_directories(markloc_regionfmt PRIVATE src modules/chardet tools)
add_executable(markloc_scratchparts tools/scratchparts.cpp)
target_link_libraries(markloc_scratchparts PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_scratchparts PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_hitviz tools/hitviz.cpp)
target_link_libraries(markloc_hitviz PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_hitviz PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_locate tools/locate.cpp)
target_link_libraries(markloc_locate PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_locate PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_charhitdetail tools/charhit_detail.cpp)
target_link_libraries(markloc_charhitdetail PRIVATE markloc_core ${OpenCV_LIBS})
target_include_directories(markloc_charhitdetail PRIVATE src modules/chardet tools)
add_executable(markloc_linediag tools/line_diag.cpp)
target_link_libraries(markloc_linediag PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_linediag PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_ringscan tools/ring_scan.cpp)
target_link_libraries(markloc_ringscan PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_ringscan PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_polar tools/polar_viz.cpp)
target_link_libraries(markloc_polar PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_polar PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_ellipsealign tools/ellipse_align.cpp)
target_link_libraries(markloc_ellipsealign PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_ellipsealign PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_hole tools/hole_test.cpp)
target_link_libraries(markloc_hole PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_hole PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_logolocate tools/logo_locate.cpp)
target_link_libraries(markloc_logolocate PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_logolocate PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_logoviz tools/logo_viz.cpp)
target_link_libraries(markloc_logoviz PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_logoviz PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_logotrack tools/logo_track.cpp)
target_link_libraries(markloc_logotrack PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_logotrack PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_logopanel tools/logo_panel.cpp)
target_link_libraries(markloc_logopanel PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_logopanel PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_guipattern tools/gui_pattern_path.cpp)
target_link_libraries(markloc_guipattern PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_guipattern PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_ellcheck tools/ellipse_sign.cpp)
target_link_libraries(markloc_ellcheck PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_ellcheck PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_innerratio tools/inner_ratio.cpp)
target_link_libraries(markloc_innerratio PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_innerratio PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_bandprobe tools/band_probe.cpp)
target_link_libraries(markloc_bandprobe PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_bandprobe PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_unsetlogo tools/unset_logo.cpp)
target_link_libraries(markloc_unsetlogo PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_unsetlogo PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_unsetone tools/unset_one.cpp)
target_link_libraries(markloc_unsetone PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_unsetone PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_alignpanel tools/align_panel.cpp)
target_link_libraries(markloc_alignpanel PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_alignpanel PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_docscoring tools/doc_scoring.cpp)
target_link_libraries(markloc_docscoring PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_docscoring PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_ringprofile tools/ring_profile.cpp)
target_link_libraries(markloc_ringprofile PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_ringprofile PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_thrscan tools/thr_scan.cpp)
target_link_libraries(markloc_thrscan PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_thrscan PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_ringpanel tools/ring_panel.cpp)
target_link_libraries(markloc_ringpanel PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_ringpanel PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_relprobe tools/relative_probe.cpp)
target_link_libraries(markloc_relprobe PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_relprobe PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_topoprobe tools/topo_probe.cpp)
target_link_libraries(markloc_topoprobe PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_topoprobe PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_featscan tools/feat_scan.cpp)
target_link_libraries(markloc_featscan PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_featscan PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_pcalogo tools/pca_logo.cpp)
target_link_libraries(markloc_pcalogo PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_pcalogo PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_pcacv tools/pca_cv.cpp)
target_link_libraries(markloc_pcacv PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_pcacv PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_batchpair tools/batch_pair.cpp)
target_link_libraries(markloc_batchpair PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_batchpair PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_colorprobe tools/color_probe.cpp)
target_link_libraries(markloc_colorprobe PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_colorprobe PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_colorspots tools/color_spots.cpp)
target_link_libraries(markloc_colorspots PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_colorspots PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_segprobe tools/seg_probe.cpp)
target_link_libraries(markloc_segprobe PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_segprobe PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_anglescan tools/angle_scan.cpp)
target_link_libraries(markloc_anglescan PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_anglescan PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_logosize tools/logo_size_probe.cpp)
target_link_libraries(markloc_logosize PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_logosize PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
if (TARGET markloc_logosize)
    list(APPEND ALL_TARGETS markloc_logosize)
endif()
# ============================================================ 定位插件 DLL
# 定位算法通过插件注入。宿主 markloc_locatebench 加载这些 DLL，
# 用同一批图、同一套指标横向比较不同定位算法。
add_library(markloc_locate_ncc SHARED plugins/locate_ncc/plugin_ncc.cpp)
target_include_directories(markloc_locate_ncc PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(markloc_locate_ncc PRIVATE ${OpenCV_LIBS})
set_target_properties(markloc_locate_ncc PROPERTIES
    PREFIX "" OUTPUT_NAME "locate_ncc"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release/plugins"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/plugins"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/plugins")
if (MSVC)
    target_compile_options(markloc_locate_ncc PRIVATE /W4 /permissive- /utf-8 /Zc:__cplusplus)
endif()
list(APPEND ALL_TARGETS markloc_locate_ncc)

add_executable(markloc_locatebench tools/locate_bench.cpp)
target_link_libraries(markloc_locatebench PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_locatebench PRIVATE src src/gui modules/chardet tools include ${Qt6_INCLUDE_DIRS})
list(APPEND ALL_TARGETS markloc_locatebench)
add_executable(markloc_scaleprobe tools/scale_probe.cpp)
target_link_libraries(markloc_scaleprobe PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_scaleprobe PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
if (TARGET markloc_scaleprobe)
    list(APPEND ALL_TARGETS markloc_scaleprobe)
endif()
add_executable(markloc_patternbench tools/pattern_bench.cpp)
target_link_libraries(markloc_patternbench PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_patternbench PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
if (TARGET markloc_patternbench)
    list(APPEND ALL_TARGETS markloc_patternbench)
endif()
add_executable(markloc_dipprobe tools/dip_probe.cpp)
target_link_libraries(markloc_dipprobe PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_dipprobe PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_closure tools/ring_closure.cpp)
target_link_libraries(markloc_closure PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_closure PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_tmplcmp tools/tmpl_cmp.cpp)
target_link_libraries(markloc_tmplcmp PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_tmplcmp PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_notch tools/notch.cpp)
target_link_libraries(markloc_notch PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_notch PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_panelhole tools/panel_hole.cpp)
target_link_libraries(markloc_panelhole PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_panelhole PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_patterncheck tools/pattern_check.cpp)
target_link_libraries(markloc_patterncheck PRIVATE markloc_core ${OpenCV_LIBS} Qt6::Core)
target_include_directories(markloc_patterncheck PRIVATE src src/gui modules/chardet tools ${Qt6_INCLUDE_DIRS})
add_executable(markloc_projalignbox tools/projalign_box.cpp)
    target_link_libraries(markloc_projalignbox PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_projalignbox PRIVATE src modules/chardet tools)

    # Projection alignment on real data with a known shift.
    add_executable(markloc_projalign tools/projalign.cpp)
    target_link_libraries(markloc_projalign PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_projalign PRIVATE src modules/chardet tools)

    # Projection alignment reliability on real parts: the peak margin is only 0.00~0.03 (the 1D profile cannot pick a unique position), so it must not be trusted as a primary aligner.
    add_executable(markloc_projreal tools/projreal.cpp)
    target_link_libraries(markloc_projreal PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_projreal PRIVATE src modules/chardet tools)

    # Direct comparison of three alignment modes: none / phase correlation / projection.
    add_executable(markloc_aligncmp tools/aligncmp.cpp)
    target_link_libraries(markloc_aligncmp PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_aligncmp PRIVATE src modules/chardet tools)

    # Coarse-fine two-stage alignment: projection coarse + phase-correlation fine, to check whether the reliable range really widens.
    add_executable(markloc_coarsefine tools/coarsefine.cpp)
    target_link_libraries(markloc_coarsefine PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_coarsefine PRIVATE src modules/chardet tools)

    # warpAffine 鏂瑰悜璇箟鏈€灏忓鐜帮紝闃叉骞崇Щ鏂瑰悜鎼炲弽銆?
    add_executable(markloc_warpsem tools/warpsem.cpp)
    target_link_libraries(markloc_warpsem PRIVATE ${OpenCV_LIBS})

    # Are char boxes fully containing glyphs: per-side overflow measurement.
    add_executable(markloc_boxwrap tools/boxwrap.cpp)
    target_link_libraries(markloc_boxwrap PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_boxwrap PRIVATE src modules/chardet tools)

    # Box containment visualisation: red box = ink overflows, green box = fully wrapped, thin yellow line = glyph bounding box.
    add_executable(markloc_boxviz tools/boxviz.cpp)
    target_link_libraries(markloc_boxviz PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_boxviz PRIVATE src modules/chardet tools)

    # Single-box ink profile: which side overflows and by how much.
    add_executable(markloc_boxpeek tools/boxpeek.cpp)
    target_link_libraries(markloc_boxpeek PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_boxpeek PRIVATE src modules/chardet tools)

    # Char-box offset robustness: in-box ink centroid as the primary criterion.
    add_executable(markloc_boxcentroid tools/boxcentroid.cpp)
    target_link_libraries(markloc_boxcentroid PRIVATE markloc_core ${OpenCV_LIBS})
    target_include_directories(markloc_boxcentroid PRIVATE src modules/chardet tools)

    # These tools were added later and were never registered in ALL_TARGETS, so they never received
    # /utf-8: CJK comments in the source get decoded as code page 936, trailing bytes swallow the
    # newline, and the reported line numbers point at the wrong place (once blamed a comment line in annotate.hpp).
    list(APPEND ALL_TARGETS markloc_diagscratch markloc_scratchcmp markloc_suspect
         markloc_grilleval markloc_grillediag markloc_annotedemo markloc_bandplot
         markloc_boxdiag markloc_scratchcmp2 markloc_scratchtrace markloc_freqvs markloc_maskeval markloc_masksweep markloc_alignlimit markloc_projcheck
         markloc_projalign markloc_projreal markloc_aligncmp markloc_coarsefine
         markloc_warpsem markloc_boxwrap markloc_boxviz markloc_boxpeek markloc_boxcentroid
         markloc_p90stat markloc_boundmgr markloc_scalenorm markloc_sigoverlay
         markloc_boxleak markloc_logostab markloc_alignprobe markloc_quick
         markloc_refcmp markloc_freqcov markloc_leakwhere markloc_normtest
         markloc_hitwhere markloc_crossrule markloc_edgeleak markloc_dotstat
         markloc_hitpaint markloc_cropzoom markloc_chain markloc_thrmatch
         markloc_fixexp markloc_trace markloc_strokefreq markloc_maskeval2
         markloc_leakratio markloc_retrain markloc_split0109 markloc_hitcount markloc_painthits markloc_inkdist markloc_gatewhy markloc_boxaudit markloc_sweepcover markloc_lineform markloc_aligndir markloc_aligncheck markloc_vsmodel markloc_boxmatch markloc_warpsign markloc_projalign markloc_projalignbox markloc_chase0101 markloc_regionalign markloc_logobreak markloc_logosep markloc_logobreak2 markloc_imgdiff markloc_cropcmp markloc_breakscan markloc_oksep markloc_bodyloc markloc_bodyline markloc_bodyproj markloc_projmatch markloc_cropfit)

    # ---- explicit moc invocation
    get_target_property(_qmake_exe Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(_qt_bin "${_qmake_exe}" DIRECTORY)
    find_program(MOC_EXE moc HINTS "${_qt_bin}" NO_DEFAULT_PATH)
    if (NOT MOC_EXE)
        find_program(MOC_EXE moc)
    endif()
    if (NOT MOC_EXE)
        message(FATAL_ERROR "moc not found, cannot build GUI")
    endif()

    set(_moc_dir "${CMAKE_CURRENT_BINARY_DIR}/moc")
    file(MAKE_DIRECTORY "${_moc_dir}")

    set(_moc_gui "")
    set(_moc_train "")
    set(_moc_split "")
    set(_moc_defect "")
    set(_moc_locateview "")
    foreach (_hdr ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/TrainWindow.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/TrainToolWindow.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/CharBoxEditor.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/SplitVerifyWindow.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/DefectViewWindow.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/BoxCanvas.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/LocateViewWindow.hpp)
        get_filename_component(_stem "${_hdr}" NAME_WE)
        set(_moc_cpp "${_moc_dir}/moc_${_stem}.cpp")
        add_custom_command(
            OUTPUT "${_moc_cpp}"
            COMMAND "${MOC_EXE}" ${_hdr} -o "${_moc_cpp}"
            DEPENDS "${_hdr}"
            COMMENT "moc ${_stem}.hpp"
            VERBATIM)
        if (_stem STREQUAL "TrainWindow")
            list(APPEND _moc_gui "${_moc_cpp}")
        elseif (_stem STREQUAL "SplitVerifyWindow")
            list(APPEND _moc_split "${_moc_cpp}")
        elseif (_stem STREQUAL "DefectViewWindow")
            list(APPEND _moc_defect "${_moc_cpp}")
        elseif (_stem STREQUAL "LocateViewWindow")
            list(APPEND _moc_locateview "${_moc_cpp}")
        elseif (_stem STREQUAL "BoxCanvas")
            # shared by markloc_train and markloc_splitcheck
            list(APPEND _moc_train "${_moc_cpp}")
            list(APPEND _moc_split "${_moc_cpp}")
        else()
            list(APPEND _moc_train "${_moc_cpp}")
        endif()
    endforeach()
    target_sources(markloc_gui PRIVATE ${_moc_gui})
    target_sources(markloc_train PRIVATE ${_moc_train})
    target_sources(markloc_splitcheck PRIVATE ${_moc_split})
    target_sources(markloc_defectview PRIVATE ${_moc_defect})
    target_sources(markloc_locateview PRIVATE ${_moc_locateview})
    target_include_directories(markloc_gui PRIVATE "${_moc_dir}")
    target_include_directories(markloc_train PRIVATE "${_moc_dir}")
    target_include_directories(markloc_splitcheck PRIVATE "${_moc_dir}")
    target_include_directories(markloc_defectview PRIVATE "${_moc_dir}")
    target_include_directories(markloc_locateview PRIVATE "${_moc_dir}")

    if (WIN32)
        target_link_libraries(markloc_gui PRIVATE Psapi)
        target_link_libraries(markloc_train PRIVATE Psapi)
        target_link_libraries(markloc_splitcheck PRIVATE Psapi)
        target_link_libraries(markloc_defectview PRIVATE Psapi)
    endif()

    message(STATUS "Qt6 version    : ${Qt6_VERSION}")
    message(STATUS "moc            : ${MOC_EXE}")
    list(APPEND ALL_TARGETS markloc_gui markloc_train markloc_splitcheck markloc_defectview)

    # ---- deploy Qt runtime next to the executables
    if (WIN32)
        find_program(WINDEPLOYQT_EXE windeployqt HINTS "${_qt_bin}")
        if (WINDEPLOYQT_EXE)
            foreach (_app markloc_gui markloc_train markloc_splitcheck markloc_defectview)
                add_custom_command(TARGET ${_app} POST_BUILD
                    COMMAND "${WINDEPLOYQT_EXE}" --no-translations --no-system-d3d-compiler
                            --no-opengl-sw "$<TARGET_FILE:${_app}>"
                    COMMENT "windeployqt: ${_app}")
            endforeach()
        endif()
    endif()
else()
    message(STATUS "Qt6 not found - GUI targets skipped")
endif()

# ============================================================ MSVC flags
if (MSVC)
    foreach (tgt ${ALL_TARGETS})
        target_compile_options(${tgt} PRIVATE /W4 /permissive- /utf-8 /Zc:__cplusplus)
        set_property(TARGET ${tgt} PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endforeach()
endif()

# ============================================================ OpenCV runtime
if (WIN32)
    set(OpenCV_DLL_DIR "${OpenCV_ROOT}/build/x64/vc16/bin")
    foreach (tgt ${ALL_TARGETS})
        if (TARGET ${tgt} AND NOT tgt STREQUAL "markloc_core")
            add_custom_command(TARGET ${tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<IF:$<CONFIG:Debug>,${OpenCV_DLL_DIR}/opencv_world4100d.dll,${OpenCV_DLL_DIR}/opencv_world4100.dll>"
                    "$<TARGET_FILE_DIR:${tgt}>")
        endif()
    endforeach()
endif()

