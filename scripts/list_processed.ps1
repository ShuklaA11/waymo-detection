Get-ChildItem "D:/waymo-detection-master/output/sjb" -Recurse -File |
  Select-Object -ExpandProperty Name |
  Where-Object { $_ -like "waymo_*_clip*" } |
  ForEach-Object { $_.Split("_clip")[0].Substring(6) } |
  Sort-Object -Unique
