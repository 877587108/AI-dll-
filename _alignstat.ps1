$exe = "build\Release\markloc_boxdiag.exe"
$rows = @()
Get-ChildItem "image\ok" -Filter "*_00.tif" | ForEach-Object {
  $out = & $exe "release_defectview\model\v00" $_.FullName 1 2>&1
  $all = ($out | Out-String)
  $m = [regex]::Match($all, '\(([-+][0-9]+\.[0-9]+), ([-+][0-9]+\.[0-9]+)\)')
  if ($m.Success) {
    $dx = [double]$m.Groups[1].Value
    $dy = [double]$m.Groups[2].Value
    $shot = if ($_.Name -match '_FT_') { "FT" } elseif ($_.Name -match '_RT_') { "RT" } else { "XX" }
    $rows += [PSCustomObject]@{
      Name = $_.Name.Substring(0,8); Shot = $shot
      DX = $dx; DY = $dy
      Mag = [math]::Round([math]::Sqrt($dx*$dx + $dy*$dy), 2)
    }
  }
}
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 "out\align_stats.csv"
"total: $($rows.Count)"
""
"=== by shot ==="
$rows | Group-Object Shot | ForEach-Object {
  $g = $_.Group
  $avgMag = ($g | Measure-Object Mag -Average).Average
  $maxMag = ($g | Measure-Object Mag -Maximum).Maximum
  $big = ($g | Where-Object { $_.Mag -gt 10 }).Count
  "{0} n={1,-3} avgMag={2,6:N2} maxMag={3,6:N2} mag>10: {4}" -f $_.Name, $g.Count, $avgMag, $maxMag, $big
}
""
"=== RT DY histogram (5px bins) ==="
$rows | Where-Object { $_.Shot -eq 'RT' } | Group-Object { [math]::Round($_.DY / 5) * 5 } | Sort-Object { [double]$_.Name } | ForEach-Object {
  "DY={0,5} : {1}" -f $_.Name, $_.Count
}
""
"=== FT DY histogram (5px bins) ==="
$rows | Where-Object { $_.Shot -eq 'FT' } | Group-Object { [math]::Round($_.DY / 5) * 5 } | Sort-Object { [double]$_.Name } | ForEach-Object {
  "DY={0,5} : {1}" -f $_.Name, $_.Count
}
