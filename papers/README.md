# Benchmark papers

The benchmark has ten cases (C1–C10), each with an easy, a medium and a hard paper. The papers are
not part of this repository: whether they may be redistributed is for their publishers to decide,
not for this repository's MIT license. To reproduce a case, download the paper from its DOI and save
it under `papers/` with the name given below. Supplementary files go in a folder next to the PDF,
named after it with `_SI` appended (for example `papers/C1/medium_10.3390_ma14185264_SI/`).

The pipeline itself needs only the PDF (`python main.py --pdf papers/C1/...pdf`). Files you put under
`papers/` are ignored by git, so they are not committed by accident.

| Case | Level | Paper | License | Save as | Supplementary files |
|---|---|---|---|---|---|
| C1 | easy | Shankar et al., *Solid State Sciences* (2016), [10.1016/j.solidstatesciences.2015.12.017](https://doi.org/10.1016/j.solidstatesciences.2015.12.017) | subscription | `C1/easy_10.1016_j.solidstatesciences.2015.12.017.pdf` | — |
| C1 | medium | Dąbrowa et al., *Materials* (2021), [10.3390/ma14185264](https://doi.org/10.3390/ma14185264) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `C1/medium_10.3390_ma14185264.pdf` | `materials-1369836-supplementary.pdf`, `materials-14-05264-s001.zip` |
| C1 | hard | Mesquita et al., *Journal of Alloys and Compounds* (2021), [10.1016/j.jallcom.2020.157975](https://doi.org/10.1016/j.jallcom.2020.157975) | subscription | `C1/hard_10.1016_j.jallcom.2020.157975.pdf` | — |
| C2 | easy | Bierman et al., *Journal of the American Chemical Society* (2025), [10.1021/jacs.5c06808](https://doi.org/10.1021/jacs.5c06808) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `C2/easy_10.1021_jacs.5c06808.pdf` | `ja5c06808_si_001.pdf` |
| C2 | medium | Miao et al., *Scientific Reports* (2016), [10.1038/srep29292](https://doi.org/10.1038/srep29292) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | `C2/medium_10.1038_srep29292.pdf` | `srep29292-s1.pdf` |
| C2 | hard | Lau et al., *Chemical Science* (2018), [10.1039/c8sc01114a](https://doi.org/10.1039/c8sc01114a) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) | `C2/hard_10.1039_c8sc01114a.pdf` | `SC-009-C8SC01114A-s001.pdf` |
| C3 | easy | Saffon-Merceron et al., *Acta Crystallographica Section E Crystallographic Communications* (2019), [10.1107/S2056989019014142](https://doi.org/10.1107/S2056989019014142) | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/legalcode) | `C3/easy_10.1107_S2056989019014142.pdf` | `e-75-01759-sup1.cif` |
| C3 | medium | Sánchez-González et al., *Materials* (2020), [10.3390/ma13081840](https://doi.org/10.3390/ma13081840) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `C3/medium_10.3390_ma13081840.pdf` | `materials-13-01840-s001.pdf` |
| C3 | hard | Sapnik et al., *Nature Communications* (2021), [10.1038/s41467-021-22218-9](https://doi.org/10.1038/s41467-021-22218-9) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | `C3/hard_10.1038_s41467-021-22218-9.pdf` | `41467_2021_22218_MOESM1_ESM.pdf`, `41467_2021_22218_MOESM2_ESM.pdf` |
| C4 | easy | Seoung et al., *Materials* (2020), [10.3390/ma13173684](https://doi.org/10.3390/ma13173684) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `C4/easy_10.3390_ma13173684.pdf` | `materials-13-03684-s001.pdf` |
| C4 | medium | Bai et al., *The Journal of Physical Chemistry C* (2023), [10.1021/acs.jpcc.3c04751](https://doi.org/10.1021/acs.jpcc.3c04751) | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) | `C4/medium_10.1021_acs.jpcc.3c04751.pdf` | `jp3c04751_si_001.pdf` |
| C4 | hard | Sushkevich et al., *Chemical Science* (2020), [10.1039/d0sc01472a](https://doi.org/10.1039/d0sc01472a) | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) | `C4/hard_10.1039_d0sc01472a.pdf` | `SC-011-D0SC01472A-s001.pdf` |
| C5 | easy | Xiao et al., *Scientific Reports* (2017), [10.1038/s41598-017-01657-9](https://doi.org/10.1038/s41598-017-01657-9) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | `C5/easy_10.1038_s41598-017-01657-9.pdf` | — |
| C5 | medium | Lee et al., *Scientific Reports* (2017), [10.1038/s41598-017-07139-2](https://doi.org/10.1038/s41598-017-07139-2) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | `C5/medium_10.1038_s41598-017-07139-2.pdf` | `41598_2017_7139_MOESM1_ESM.pdf` |
| C5 | hard | Zou et al., *Nature Communications* (2019), [10.1038/s41467-019-11299-2](https://doi.org/10.1038/s41467-019-11299-2) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | `C5/hard_10.1038_s41467-019-11299-2.pdf` | `41467_2019_11299_MOESM1_ESM.pdf` |
| C6 | easy | Liu et al., *IUCrData* (2023), [10.1107/S2414314623010659](https://doi.org/10.1107/S2414314623010659) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) | `C6/easy_10.1107_S2414314623010659.pdf` | `x-08-x231065-Isup2.hkl`, `x-08-x231065-sup1.cif`, `x-08-x231065-sup3.docx` |
| C6 | medium | Feuerbacher et al., *Scripta Materialia* (2018), [10.1016/j.scriptamat.2018.04.009](https://doi.org/10.1016/j.scriptamat.2018.04.009) | subscription | `C6/medium_10.1016_j.scriptamat.2018.04.009.pdf` | — |
| C6 | hard | Hsiao et al., *Nature Communications* (2022), [10.1038/s41467-022-34335-0](https://doi.org/10.1038/s41467-022-34335-0) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | `C6/hard_10.1038_s41467-022-34335-0.pdf` | `41467_2022_34335_MOESM1_ESM.pdf`, `41467_2022_34335_MOESM2_ESM.pdf` |
| C7 | easy | Sun et al., *National Science Review* (2020), [10.1093/nsr/nwaa195](https://doi.org/10.1093/nsr/nwaa195) | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) | `C7/easy_10.1093_nsr_nwaa195.pdf` | `Ni-TPP.cif`, `Ni-TPYP-1.cif`, `Ni-TPYP.cif`, `Revised Supporting Information.doc` |
| C7 | medium | Xue et al., *Nanoscale* (2020), [10.1039/d0nr06006b](https://doi.org/10.1039/d0nr06006b) | subscription | `C7/medium_10.1039_d0nr06006b.pdf` | — |
| C7 | hard | Yang et al., *Nature Communications* (2019), [10.1038/s41467-019-12510-0](https://doi.org/10.1038/s41467-019-12510-0) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | `C7/hard_10.1038_s41467-019-12510-0.pdf` | `41467_2019_12510_MOESM1_ESM.pdf` |
| C8 | easy | Costley-Wood et al., *Chemistry of Materials* (2025), [10.1021/acs.chemmater.5c01417](https://doi.org/10.1021/acs.chemmater.5c01417) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `C8/easy_10.1021_acs.chemmater.5c01417.pdf` | `cm5c01417_si_001.pdf` |
| C8 | medium | Zieliński et al., *Scientific Reports* (2023), [10.1038/s41598-023-28557-5](https://doi.org/10.1038/s41598-023-28557-5) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | `C8/medium_10.1038_s41598-023-28557-5.pdf` | `41598_2023_28557_MOESM1_ESM.pdf` |
| C8 | hard | Tan et al., *Nature Communications* (2022), [10.1038/s41467-022-34797-2](https://doi.org/10.1038/s41467-022-34797-2) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | `C8/hard_10.1038_s41467-022-34797-2.pdf` | `41467_2022_34797_MOESM1_ESM.pdf` |
| C9 | easy | Robben et al., *Acta Crystallographica Section E Crystallographic Communications* (2016), [10.1107/S2056989016001924](https://doi.org/10.1107/S2056989016001924) | [CC BY 2.0 UK](http://creativecommons.org/licenses/by/2.0/uk/legalcode) | `C9/easy_10.1107_S2056989016001924.pdf` | `e-72-00287-Isup2.hkl`, `e-72-00287-sup1.cif` |
| C9 | medium | Rusdi et al., *Materials* (2021), [10.3390/ma14237342](https://doi.org/10.3390/ma14237342) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `C9/medium_10.3390_ma14237342.pdf` | — |
| C9 | hard | Shiotani et al., *Scientific Reports* (2017), [10.1038/s41598-017-07086-y](https://doi.org/10.1038/s41598-017-07086-y) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | `C9/hard_10.1038_s41598-017-07086-y.pdf` | — |
| C10 | easy | Hammer et al., arXiv (2026), [arXiv:2610.23585](https://arxiv.org/abs/2610.23585) | arXiv (non-exclusive) | `C10/easy_arxiv_2610.23585.pdf` | — |
| C10 | medium | Aitchison et al., *Physical Chemistry Chemical Physics* (2017), [10.1039/c7cp04244b](https://doi.org/10.1039/c7cp04244b) | subscription | `C10/medium_10.1039_c7cp04244b.pdf` | — |
| C10 | hard | Behjati and Koper, *The Journal of Physical Chemistry C* (2024), [10.1021/acs.jpcc.4c06362](https://doi.org/10.1021/acs.jpcc.4c06362) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `C10/hard_10.1021_acs.jpcc.4c06362.pdf` | `jp4c06362_si_001.pdf` |
