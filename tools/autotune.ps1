# 自动调参（爬山/坐标轮换）——对单个视角在给定样本集上搜索更优参数
# 用法: .\autotune.ps1 -View 02 -ScanDir out\scan_v02 -Exe .\build\Release\markloc_gui.exe -ModelDir models\char
param(
  [Parameter(Mandatory=$true)][string]$View,
  [Parameter(Mandatory=$true)][string]$ScanDir,
  [string]$Exe = ".\build\Release\markloc_gui.exe",
  [string]$ModelDir = "models\char",
  [int]$MaxEvals = 40
)

$paramFile = Join-Path $ModelDir "v$View\params.txt"

function Get-Line($path) { (Get-Content $path)[0] }
function Set-Field($path, $idx, $val) {
  $lines = [System.IO.File]::ReadAllLines($path)
  $f = $lines[0] -split '\s+'
  $f[$idx] = "$val"
  $lines[0] = $f -join ' '
  [System.IO.File]::WriteAllLines($path, $lines, [System.Text.UTF8Encoding]::new($false))
}

function Evaluate {
  $out = & $Exe --batch $ScanDir 2>&1 | Out-String
  $mm = [regex]::Matches($out, 'TP=(\d+)\s+FP=(\d+)\s+FN=(\d+)\s+TN=(\d+)')
  if ($mm.Count -eq 0) { return @{ TP=-1; FP=999; Obj=-9999 } }
  $m = $mm[$mm.Count-1]
  $tp = [int]$m.Groups[1].Value
  $fp = [int]$m.Groups[2].Value
  # 目标：召回尽量高、误报重罚
  $obj = $tp - 5 * $fp
  return @{ TP=$tp; FP=$fp; Obj=$obj }
}

# 参数表：(索引, 名称, 候选值)
$grid = @(
  @(0,  'sigma',        @(12, 15, 20, 25, 35)),
  @(1,  'strokeHi',     @(0.28, 0.35, 0.42, 0.50)),
  @(2,  'strokeLo',     @(0.15, 0.20, 0.26)),
  @(3,  'pKeep',        @(0.15, 0.20, 0.28)),
  @(4,  'dilateK',      @(7, 9, 13)),
  @(11, 'scratchMinLen',@(100, 130, 160, 200)),
  @(10, 'scratchCov',   @(0.15, 0.20, 0.25, 0.35))
)

$evals = 0
$base = Evaluate
$evals++
$best = -5 * $base['FP'] + $base['TP']
Write-Host ("[起始] " + (Get-Line $paramFile))
Write-Host ("        TP=$($base['TP']) FP=$($base['FP']) obj=$best")

foreach ($round in 1..2) {
  $improved = $false
  foreach ($g in $grid) {
    if ($evals -ge $MaxEvals) { break }
    $idx = $g[0]; $name = $g[1]; $cands = $g[2]
    $cur = (Get-Line $paramFile) -split '\s+'
    $curVal = $cur[$idx]
    $bestVal = $curVal
    foreach ($c in $cands) {
      if ("$c" -eq "$curVal") { continue }
      if ($evals -ge $MaxEvals) { break }
      Set-Field $paramFile $idx $c
      $r = Evaluate
      $evals++
      if ((-5 * $r['FP'] + $r['TP']) -gt $best) {
        $best = -5 * $r['FP'] + $r['TP']; $bestVal = "$c"; $improved = $true
        Write-Host ("  [更优] $name=$c  TP=$($r['TP']) FP=$($r['FP']) obj=$best  (第$evals次)")
      }
    }
    Set-Field $paramFile $idx $bestVal
  }
  if (-not $improved) { break }
}

# 依据最终 OK/NG 分布设定阈值：使 OK 最大值 ≈0.85
$out = & $Exe --batch $ScanDir 2>&1 | Out-String
$okMax = 0.0; $ngMin = 99.0
foreach ($l in ($out -split "`r?`n")) {
  if ($l -match '^\s+(\d{4})\s+(OK|NG)\s+(OK|NG)\s+([\d.]+)\s+\|\s+(.*)$') {
    $lab = $matches[2]
    $vals = $matches[5] -split '\s+' | Where-Object { $_ -ne '' }
    $v = [double]$vals[[int]$View]
    if ($lab -eq 'OK') { if ($v -gt $okMax) { $okMax = $v } }
    else { if ($v -lt $ngMin) { $ngMin = $v } }
  }
}
$curT = [double]((Get-Line $paramFile) -split '\s+')[7]
if ($okMax -gt 0.01) {
  $newT = [int]($curT * $okMax / 0.85)
  Set-Field $paramFile 7 $newT
  Write-Host "  阈值调整: scratchT $curT -> $newT (OK最大值 $([math]::Round($okMax,2)))"
}

$final = Evaluate
Write-Host ("[完成] " + (Get-Line $paramFile))
Write-Host ("        TP=$($final['TP']) FP=$($final['FP']) obj=$(-5 * $final['FP'] + $final['TP'])  评估次数=$evals")
