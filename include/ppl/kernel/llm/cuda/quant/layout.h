# include <assert.h>
# include "common.h"

# ifndef __PPL_KERNEL_LLM_CUDA_QUANT_LAYOUT_H__
# define __PPL_KERNEL_LLM_CUDA_QUANT_LAYOUT_H__

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace quant {

// CUDA Data Layout Explanation

// Contiguous Memory Layout (Row-Major):
// - Data elements are stored in adjacent memory locations.
// - Similar to traditional CPU memory, where elements in a row are stored sequentially.
// - Efficient when neighboring threads access neighboring elements in parallel,
//   minimizing memory access latency.
// - Suitable when data needs to be accessed in a row-wise or element-wise manner.
// - Example: Suitable for 2D matrices when elements in the same row are frequently accessed together.

// Strided Memory Layout (Column-Major):
// - Data elements are stored with a specified stride between them.
// - Useful when data access patterns are irregular, and threads access non-adjacent elements.
// - Allows fine-grained control of memory access patterns.
// - Suitable when different threads need to access different columns or non-adjacent elements.
// - Accessing data in this layout is efficient for non-adjacent element access.

// By ChatGPT

struct MatrixCoord{
    using index_t = int32_t;
    index_t row_id;
    index_t col_id;

    MatrixCoord() = delete;
    MatrixCoord(index_t row_id, index_t col_id): 
        row_id(row_id), col_id(col_id) {
        // constructor
    }
}

/*
    Layout Helper that convert row major index to col32 index.
*/
struct LayoutConverter {
    using index_t = unsigned int;
    index_t num_of_row; // num of row of given matrix.
    index_t num_of_col; // num of col of given matrix.

    LayoutConverter() = delete; // 别瞎几把初始化
    
    __HOST_DEVICE_FUNCTION__
    LayoutConverter(index_t num_of_row, index_t num_of_col): 
        num_of_row(num_of_row), num_of_col(num_of_col) {
        // constructor
    };

    const __HOST_DEVICE_FUNCTION__
    inline index_t RowColToOffset(
        const index_t row, const index_t col){
        return row * num_of_col + col;
    }

    const __HOST_DEVICE_FUNCTION__
    inline index_t RowMajorToCol32(const index_t offset){
        assert (offset < num_of_col * num_of_col);
        constexpr index_t COL32 = 32;
        
        index_t row = offset / COL32;                            // row in col32 layout
        index_t col = offset % COL32 + row / num_of_row * COL32; // col in col32 layout
        return RowColToOffset(row, col);
    }

    const __HOST_DEVICE_FUNCTION__
    inline index_t RowMajorToCol32(
        const index_t row, const index_t col){
        assert (row < num_of_row);
        assert (col < num_of_col);
        return RowMajorToCol32(RowColToOffset(row, col));
    }

    const __HOST_DEVICE_FUNCTION__
    inline index_t Col32ToRowMajor(
        const index_t offset){
        assert (offset < num_of_col * num_of_col);
        return Col32ToRowMajor(offset / num_of_col, offset % num_of_col);
    }

    const __HOST_DEVICE_FUNCTION__
    inline index_t Col32ToRowMajor(
        const index_t row, const index_t col){
        constexpr index_t COL32 = 32;
        assert (row < num_of_row);
        assert (col < num_of_col);

        index_t col_block_id = col / COL32;
        index_t internal_id = col_block_id - col_block_id * COL32;
        return col_block_id * num_of_row * COL32 + row * COL32 + internal_id;
    }

    const __HOST_DEVICE_FUNCTION__
    inline MatrixCoord Col32ToRowMajorCoord(
        const index_t row, const index_t col){
        constexpr index_t COL32 = 32;
        assert (row < num_of_row);
        assert (col < num_of_col);

        index_t col_block_id = col / COL32;
        index_t internal_id = col_block_id - col_block_id * COL32;
        index_t row_major_offset = col_block_id * num_of_row * COL32 + row * COL32 + internal_id;
        return MatrixCoord(row_major_offset / num_of_col, row_major_offset % num_of_col);
    }

    const __HOST_DEVICE_FUNCTION__
    inline MatrixCoord Col32ToRowMajorCoord(
        const index_t offset){
        assert (offset < num_of_col * num_of_col);
        return Col32ToRowMajorCoord(offset / num_of_col, offset % num_of_col);
    }
};

}}}}}

# endif