$ErrorActionPreference = 'Stop'
$docPath = 'C:\project\MMDL\reports\math13_occlusion_report_20261014.docx'
$pdfPath = 'C:\project\MMDL\reports\math13_occlusion_report_qa\word_render.pdf'
$wordInstance = $null
$reportDoc = $null
try {
    $wordInstance = New-Object -ComObject Word.Application
    $wordInstance.Visible = $false
    $wordInstance.DisplayAlerts = 0
    $reportDoc = $wordInstance.Documents.Open($docPath, $false, $true)
    $reportDoc.Repaginate()
    $pageCount = $reportDoc.ComputeStatistics(2)
    $reportDoc.ExportAsFixedFormat($pdfPath, 17)
    Write-Output "Rendered pages: $pageCount"
    Write-Output $pdfPath
} finally {
    if ($null -ne $reportDoc) {
        $reportDoc.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($reportDoc)
    }
    if ($null -ne $wordInstance) {
        $wordInstance.Quit(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordInstance)
    }
}
