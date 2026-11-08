# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Extracted unchanged from the pinned source; host evidence below is not executable.
import triton
import triton.language as tl

@triton.jit
def _embedding_dense_backward_count_kernel(
    indices_ptr,
    counts_ptr,
    N,
    num_weights,
    padding_idx,
    BLOCK_N: tl.constexpr,
    total_blocks: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)

    for block_id in range(pid, total_blocks, num_jobs):
        offs = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        idx = tl.load(indices_ptr + offs, mask=mask, other=0).to(tl.int32)
        valid = mask & (idx != padding_idx) & (idx >= 0) & (idx < num_weights)
        tl.atomic_add(counts_ptr + idx, 1, mask=valid)

