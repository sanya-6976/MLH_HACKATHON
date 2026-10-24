# Real Dataset Audit: Food Packaging OCR Dataset

## Provenance and use

- Dataset name: `Food Packaging OCR Dataset`, as named by the supplied README and ZIP.
- Source URL, authors/publisher, version/release, and retrieval date: **UNKNOWN**. The README and archive contain no source page, author, publisher, version, or retrieval metadata.
- Licence, permitted use, attribution requirements, and redistribution restrictions: **UNKNOWN**. No licence or terms file is present in the archive or extracted tree. A public or locally supplied ZIP does not establish permission. Confirm the original source page and terms before showing these images in a hackathon demo, publishing them, or redistributing images/annotations. Until confirmed, keep use local and do not redistribute.
- The README does not state any demo-specific permission or restriction. No legal conclusion about use can be made from these files alone.

## Verified contents

The ZIP contains `Food Packaging OCR Dataset/README.txt` and two task folders. Each task has `train/`, `valid/`, and `test/`; each split has `images/` and `labels.txt`. The archive contains JPEG (`.jpg`) images and those six annotation files; no separate codebook, metadata, or licence file was found.

```text
Food Packaging OCR Dataset/
  README.txt
  det/{train,valid,test}/{images/*.jpg,labels.txt}
  rec/{train,valid,test}/{images/*.jpg,labels.txt}
```

The README describes `det` as text detection and `rec` as text recognition. Its stated counts (verified against ZIP image entries) are 8,736 train, 1,092 validation, and 1,092 test images per task. The local extraction is not complete: `det` has all three stated splits; `rec/test` has 1,092 images, `rec/train` is partial (4,267 JPEGs at audit time), and `rec/valid` is missing. The ZIP itself includes all three `rec` splits at the stated counts.

## Annotation formats and scope

The README specifies tab-separated records. Detection rows contain an image filename and a JSON list of regions, each with `transcription` and `points`. For example, the first region for `det/test/0001.jpg` is:

```json
{"transcription":"UDDER","points":[[152,127],[212,127],[212,181],[152,181]]}
```

The actual row includes additional regions such as `CHOCOLUSH`, `MILKSHAKE`, and `made`. The README calls `points` polygon points but does not document coordinate ordering or a more detailed geometry convention.

Recognition rows contain an image filename and a transcription, for example `rec/test/0001.jpg<TAB>REDBUSH`. The image is a narrow text crop. `rec/test/0003.jpg` is labeled `Whitworths` and shows stylized script lettering.

These annotations support text-region detection/localization with transcribed text for `det`, and transcription for the cropped text samples in `rec`. They do **not** explicitly annotate ingredient lists, nutrition values, allergens, expiry dates, product fields, or regulatory status. Text that happens to mention a nutrition panel or ingredient on a package is still only a generic text-region annotation. The OCR labels are not ground truth for LabelGuard fields or regulatory compliance.

## Selected subset

The separate manifest at [real_image_selection.json](real_image_selection.json) references 10 existing, opened images from the official test split: eight `det/test` package photos and two `rec/test` text crops. The detection examples vary in product/package shape, layout, label contrast, and annotation density (8 to 31 regions in the sampled rows); recognition includes a simple uppercase crop and a stylized wordmark. The local files were checked for existence and opened; selected detection image dimensions range from 159x400 to 400x314 pixels.

Detection expected ground truth is referenced by image row in the original `labels.txt`, avoiding a duplicate of the full polygon annotations. Recognition transcriptions are included directly from their label rows. All ten records are test split samples; do not add training images to reported evaluation. This small, manually chosen subset is useful for smoke testing only and is not representative enough for broad performance claims.

## What this dataset can measure

- Separately, text detection/localization on the selected `det/test` photos, using the provided text and polygon annotations and a stated matching protocol.
- Separately, text transcription on the selected `rec/test` crops, using the provided transcription strings and a stated text normalization protocol.
- Qualitative handling of varied photographed packaging and cropped text, with annotation coverage and uncertainty stated.

It cannot establish accuracy for LabelGuard's product-specific fields, completeness of ingredient/nutrition/allergen extraction, expiry-date extraction, FSSAI compliance, or the correctness of any rules. The source does not supply semantic field mappings. No accuracy was calculated, and no rule is verified by these labels. The dataset is not established as India-specific; sample packages include international brands and labeling conventions. OCR annotations have not been independently adjudicated against the images.

## Remaining checks

Find the original dataset source and review its licence/terms, attribution instructions, redistribution rules, and release/version. Confirm whether demo display is permitted. Recover the missing local recognition validation split and remaining training images only if needed and permitted. These provenance and usage questions block any claim that public/demo redistribution is cleared.