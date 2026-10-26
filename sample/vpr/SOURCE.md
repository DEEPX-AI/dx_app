# sample/vpr -- Visual Place Recognition toy dataset

Fetched by `scripts/fetch_vpr_toy_dataset.py` from
<https://github.com/gmberton/VPR-methods-evaluation/tree/master/toy_dataset>.

* `database/` -- 17 images, the places to be recognised.
* `queries/`  -- 5 images, the same places from a different viewpoint.

That repo is the standardised VPR evaluation codebase referenced by the EigenPlaces
README; this is the toy set it ships for a quick unlabelled run.

**No ground truth.** The filenames carry no `@utm_east@utm_north@` fields, so top-k
retrieval is demonstrable but Recall@k is not computable. For a geo-referenced set use
<https://github.com/gmberton/VPR-datasets-downloader> and point the gallery builder at
`datasets/<name>/images/test/database`.

## Upstream licence (gmberton/VPR-methods-evaluation)

```
MIT License

Copyright (c) 2023 Gabriele Berton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
