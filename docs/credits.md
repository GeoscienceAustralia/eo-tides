## Citing `eo-tides`

To cite `eo-tides` in your work, please use the following [Journal of Open Source Software](https://doi.org/10.21105/joss.07786) citation:

=== "Plain text"

    ```
    Bishop-Taylor, R., Phillips, C., Sagar, S., Newey, V., & Sutterley, T., (2025). eo-tides: Tide modelling tools for large-scale satellite Earth observation analysis. Journal of Open Source Software, 10(109), 7786, https://doi.org/10.21105/joss.07786
    ```

=== "BibTeX"

    ```
    @article{Bishop-Taylor2025,
    doi       = {10.21105/joss.07786},
    url       = {https://doi.org/10.21105/joss.07786},
    year      = {2025},
    publisher = {The Open Journal},
    volume    = {10},
    number    = {109},
    pages     = {7786},
    author    = {Robbi Bishop-Taylor and Claire Phillips and Stephen Sagar and Vanessa Newey and Tyler Sutterley},
    title     = {eo-tides: Tide modelling tools for large-scale satellite Earth observation analysis},
    journal   = {Journal of Open Source Software}
    }
    ```

In addition, please consider also citing the underlying [`pyTMD` Python package](https://pytmd.readthedocs.io/en/latest/) which powers the tide modelling functionality behind `eo-tides`:

```
Sutterley, T. C., Alley, K., Brunt, K., Howard, S., Padman, L., Siegfried, M. (2017) pyTMD: Python-based tidal prediction software. 10.5281/zenodo.5555395
```

## Credits

`eo-tides` builds on (and wouldn't be possible without!) fundamental tide modelling tools provided by `pyTMD`. The authors wish to thank Dr. Tyler Sutterley for his ongoing development and support of this incredible modelling tool.

Functions from `eo-tides` were originally developed in the [`Digital Earth Australia Notebooks and Tools` repository](https://github.com/GeoscienceAustralia/dea-notebooks/). The authors would like to thank all DEA Notebooks contributors and maintainers for their invaluable assistance with code review, feature suggestions and code edits.

FES Finite Element Solution tide models were developed, validated by the CTOH/LEGOS, France and distributed by Aviso+: <https://www.aviso.altimetry.fr/en/data/products/sea-surface-height-products/regional/x-track-sla/x-track-l2p-sla-version-2022.html>

This repository was initialised using the [`cookiecutter-uv`](https://github.com/fpgmaas/cookiecutter-uv) package.

## References

<small>

Carrere et al., OSTST 2022: A new barotropic tide model for global ocean: FES2022,
<https://doi.org/10.24400/527896/a03-2022.3287>

Egbert, Gary D., and Svetlana Y. Erofeeva. "Efficient inverse modeling of barotropic ocean tides." Journal of Atmospheric and Oceanic Technology 19.2 (2002): 183-204.

Florent H. Lyard, Loren Carrere, Ergane Fouchet, Mathilde Cancet, David Greenberg, Gérald Dibarboure and Nicolas Picot: “FES2022 a step towards a SWOT-compliant tidal correction”, to be submitted to Ocean Sciences.

Hart-Davis Michael, Piccioni Gaia, Dettmering Denise, Schwatke Christian, Passaro Marcello, Seitz Florian (2021). EOT20 - A global Empirical Ocean Tide model from multi-mission satellite altimetry. SEANOE. <https://doi.org/10.17882/79489>

Hart-Davis Michael G., Piccioni Gaia, Dettmering Denise, Schwatke Christian, Passaro Marcello, Seitz Florian (2021). EOT20: a global ocean tide model from multi-mission satellite altimetry. Earth System Science Data, 13 (8), 3869-3884. <https://doi.org/10.5194/essd-13-3869-2021>

Sutterley, T. C., Markus, T., Neumann, T. A., van den Broeke, M., van Wessem, J. M., and Ligtenberg, S. R. M.: Antarctic ice shelf thickness change from multimission lidar mapping, The Cryosphere, 13, 1801–1817, https://doi.org/10.5194/tc-13-1801-2019, 2019.

</small>
