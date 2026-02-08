$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$wekaJar = Join-Path $root "weka.jar"
$modelPath = Join-Path $root "Atian_CIS304_SmokingDataset.model"
$arffPath = Join-Path $root "Atian_CIS304_SmokingDataset.arff"
$outPath = Join-Path $root "web\model.json"
$src = Join-Path $PSScriptRoot "ExportWekaNaiveBayes.java"
$build = Join-Path $PSScriptRoot "build"

New-Item -ItemType Directory -Force -Path $build | Out-Null

& javac -cp $wekaJar -d $build $src
if ($LASTEXITCODE -ne 0) { throw "javac failed" }

& java -cp "$build;$wekaJar" ExportWekaNaiveBayes $modelPath $arffPath $outPath
if ($LASTEXITCODE -ne 0) { throw "java failed" }
