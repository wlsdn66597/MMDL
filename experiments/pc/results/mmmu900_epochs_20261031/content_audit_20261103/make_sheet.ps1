param([Parameter(Mandatory=$true)][string]$Ids,[Parameter(Mandatory=$true)][string]$Out,[int]$CellW=850,[int]$CellH=620)
Add-Type -AssemblyName System.Drawing
$imgDir='C:\project\MMDL\results\mmmu_first_baseline_20261009\analysis\images'
$idx=Get-Content -LiteralPath (Join-Path $imgDir 'index.json') -Raw -Encoding utf8 | ConvertFrom-Json
$items=@(foreach($id in $Ids.Split(',')){foreach($im in $idx.$id){[pscustomobject]@{Id=$id;File=$im.file;Field=$im.field}}})
$cols=2;$rows=[int][Math]::Ceiling($items.Count/2.0)
$bmp=[System.Drawing.Bitmap]::new(($CellW*$cols),($CellH*$rows));$g=[System.Drawing.Graphics]::FromImage($bmp)
$g.Clear([System.Drawing.Color]::White);$g.InterpolationMode=[System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$font=[System.Drawing.Font]::new('Arial',15)
for($j=0;$j -lt $items.Count;$j++){
  $im=[System.Drawing.Image]::FromFile((Join-Path $imgDir $items[$j].File));$x=($j%2)*$CellW;$y=[int][Math]::Floor($j/2.0)*$CellH
  $g.DrawString(($items[$j].Id+' '+$items[$j].Field),$font,[System.Drawing.Brushes]::Black,$x+5,$y+5)
  $ratio=[Math]::Min(($CellW-16)/[double]$im.Width,($CellH-45)/[double]$im.Height)
  $w=[int]($im.Width*$ratio);$hh=[int]($im.Height*$ratio)
  $g.DrawImage($im,($x+8),($y+35),$w,$hh);$im.Dispose()
}
$dir=Split-Path -Parent $Out;if(-not(Test-Path -LiteralPath $dir)){New-Item -ItemType Directory -Path $dir -Force | Out-Null}
$bmp.Save($Out,[System.Drawing.Imaging.ImageFormat]::Png);$font.Dispose();$g.Dispose();$bmp.Dispose();Write-Output $Out
