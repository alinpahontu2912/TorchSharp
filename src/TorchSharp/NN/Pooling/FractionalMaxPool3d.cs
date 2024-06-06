// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a FractionalMaxPool3d module.
        /// </summary>
        public sealed class FractionalMaxPool3d : ParamLessModule<Tensor, Tensor>
        {
            internal FractionalMaxPool3d(long[] kernel_size, long[] output_size = null, double[] output_ratio = null) : base(nameof(FractionalMaxPool3d))
            {
                this.kernel_size = kernel_size;
                this.output_size = output_size;
                this.output_ratio = output_ratio;
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.fractional_max_pool3d(input, kernel_size, output_size, output_ratio);
            }

            public (Tensor Values, Tensor Indices) forward_with_indices(Tensor input)
            {
                return torch.nn.functional.fractional_max_pool3d_with_indices(input, kernel_size, output_size, output_ratio);
            }

            public long[] kernel_size { get; set; }
            public long[] output_size { get; set; }
            public double[] output_ratio { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies a 3d fractional max pooling over an input signal composed of several input planes.
            ///
            /// Fractional MaxPooling is described in detail in the paper Fractional MaxPooling by Ben Graham,
            /// see: https://arxiv.org/abs/1412.6071
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="output_size">The target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH</param>
            /// <param name="output_ratio">If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)</param>
            /// <returns></returns>
            public static FractionalMaxPool3d FractionalMaxPool3d(long kernel_size, long? output_size = null, double? output_ratio = null)
            {
                var pSize = output_size.HasValue ? new long[] { output_size.Value, output_size.Value, output_size.Value } : null;
                var pRatio = output_ratio.HasValue ? new double[] { output_ratio.Value, output_ratio.Value, output_ratio.Value } : null;
                return FractionalMaxPool3d(new long[] { kernel_size, kernel_size, kernel_size }, pSize, pRatio);
            }

            /// <summary>
            /// Applies a 3d fractional max pooling over an input signal composed of several input planes.
            ///
            /// Fractional MaxPooling is described in detail in the paper Fractional MaxPooling by Ben Graham,
            /// see: https://arxiv.org/abs/1412.6071
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="output_size">The target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH</param>
            /// <param name="output_ratio">If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)</param>
            /// <returns></returns>
            public static FractionalMaxPool3d FractionalMaxPool3d((long, long, long) kernel_size, (long, long, long)? output_size = null, (double, double, double)? output_ratio = null)
            {
                var pSize = output_size.HasValue ? new long[] { output_size.Value.Item1, output_size.Value.Item2, output_size.Value.Item3 } : null;
                var pRatio = output_ratio.HasValue ? new double[] { output_ratio.Value.Item1, output_ratio.Value.Item2, output_ratio.Value.Item3 } : null;
                return FractionalMaxPool3d(new long[] { kernel_size.Item1, kernel_size.Item2, kernel_size.Item3 }, pSize, pRatio);
            }

            /// <summary>
            /// Applies a 3d fractional max pooling over an input signal composed of several input planes.
            ///
            /// Fractional MaxPooling is described in detail in the paper Fractional MaxPooling by Ben Graham,
            /// see: https://arxiv.org/abs/1412.6071
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="output_size">The target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH</param>
            /// <param name="output_ratio">If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)</param>
            /// <returns></returns>
            public static FractionalMaxPool3d FractionalMaxPool3d(long[] kernel_size, long[] output_size = null, double[] output_ratio = null)
            {
                if (kernel_size == null || kernel_size.Length != 3)
                    throw new ArgumentException("Kernel size must contain three elements.");
                if (output_size != null && output_size.Length != 3)
                    throw new ArgumentException("output_size must contain three elements.");
                if (output_ratio != null && output_ratio.Length != 3)
                    throw new ArgumentException("output_ratio must contain three elements.");
                if (output_size == null && output_ratio == null)
                    throw new ArgumentNullException("Only one of output_size and output_ratio may be specified.");
                if (output_size != null && output_ratio != null)
                    throw new ArgumentNullException("FractionalMaxPool3d requires specifying either an output size, or a pooling ratio.");

                return new FractionalMaxPool3d(kernel_size, output_size, output_ratio);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 3d fractional max pooling over an input signal composed of several input planes.
                ///
                /// Fractional MaxPooling is described in detail in the paper Fractional MaxPooling by Ben Graham,
                /// see: https://arxiv.org/abs/1412.6071
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
                /// <param name="output_size">The target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH</param>
                /// <param name="output_ratio">If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)</param>
                /// <returns></returns>
                public static Tensor fractional_max_pool3d(Tensor input, long kernel_size, long? output_size = null, double? output_ratio = null)
                {
                    var pSize = output_size.HasValue ? new long[] { output_size.Value, output_size.Value, output_size.Value } : null;
                    var pRatio = output_ratio.HasValue ? new double[] { output_ratio.Value, output_ratio.Value, output_ratio.Value } : null;
                    return fractional_max_pool3d(input, new long[] { kernel_size, kernel_size, kernel_size }, pSize, pRatio);
                }

                /// <summary>
                /// Applies a 3d fractional max pooling over an input signal composed of several input planes.
                ///
                /// Fractional MaxPooling is described in detail in the paper Fractional MaxPooling by Ben Graham,
                /// see: https://arxiv.org/abs/1412.6071
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
                /// <param name="output_size">The target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH</param>
                /// <param name="output_ratio">If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)</param>
                /// <returns></returns>
                public static Tensor fractional_max_pool3d(Tensor input, (long, long, long) kernel_size, (long, long, long)? output_size = null, (double, double, double)? output_ratio = null)
                {
                    var pSize = output_size.HasValue ? new long[] { output_size.Value.Item1, output_size.Value.Item2, output_size.Value.Item3 } : null;
                    var pRatio = output_ratio.HasValue ? new double[] { output_ratio.Value.Item1, output_ratio.Value.Item2, output_ratio.Value.Item3 } : null;
                    return fractional_max_pool3d(input, new long[] { kernel_size.Item1, kernel_size.Item2, kernel_size.Item3 }, pSize, pRatio);
                }

                /// <summary>
                /// Applies a 3d fractional max pooling over an input signal composed of several input planes.
                ///
                /// Fractional MaxPooling is described in detail in the paper Fractional MaxPooling by Ben Graham,
                /// see: https://arxiv.org/abs/1412.6071
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
                /// <param name="output_size">The target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH</param>
                /// <param name="output_ratio">If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)</param>
                /// <returns></returns>
                public static Tensor fractional_max_pool3d(Tensor input, long[] kernel_size, long[] output_size = null, double[] output_ratio = null)
                {
                    var ret = fractional_max_pool3d_with_indices(input, kernel_size, output_size, output_ratio);
                    ret.Indices.Dispose();
                    return ret.Values;
                }

                /// <summary>
                /// Applies a 3d fractional max pooling over an input signal composed of several input planes.
                ///
                /// Fractional MaxPooling is described in detail in the paper Fractional MaxPooling by Ben Graham,
                /// see: https://arxiv.org/abs/1412.6071
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
                /// <param name="output_size">The target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH</param>
                /// <param name="output_ratio">If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)</param>
                /// <returns></returns>
                public static (Tensor Values, Tensor Indices) fractional_max_pool3d_with_indices(Tensor input, long kernel_size, long? output_size = null, double? output_ratio = null)
                {
                    var pSize = output_size.HasValue ? new long[] { output_size.Value, output_size.Value, output_size.Value } : null;
                    var pRatio = output_ratio.HasValue ? new double[] { output_ratio.Value, output_ratio.Value, output_ratio.Value } : null;
                    return fractional_max_pool3d_with_indices(input, new long[] { kernel_size, kernel_size, kernel_size }, pSize, pRatio);
                }

                /// <summary>
                /// Applies a 3d fractional max pooling over an input signal composed of several input planes.
                ///
                /// Fractional MaxPooling is described in detail in the paper Fractional MaxPooling by Ben Graham,
                /// see: https://arxiv.org/abs/1412.6071
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
                /// <param name="output_size">The target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH</param>
                /// <param name="output_ratio">If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)</param>
                /// <returns></returns>
                public static (Tensor Values, Tensor Indices) fractional_max_pool3d_with_indices(Tensor input, (long, long, long) kernel_size, (long, long, long)? output_size = null, (double, double, double)? output_ratio = null)
                {
                    var pSize = output_size.HasValue ? new long[] { output_size.Value.Item1, output_size.Value.Item2, output_size.Value.Item3 } : null;
                    var pRatio = output_ratio.HasValue ? new double[] { output_ratio.Value.Item1, output_ratio.Value.Item2, output_ratio.Value.Item3 } : null;
                    return fractional_max_pool3d_with_indices(input, new long[] { kernel_size.Item1, kernel_size.Item2, kernel_size.Item3 }, pSize, pRatio);
                }

                /// <summary>
                /// Applies a 3d fractional max pooling over an input signal composed of several input planes.
                ///
                /// Fractional MaxPooling is described in detail in the paper Fractional MaxPooling by Ben Graham,
                /// see: https://arxiv.org/abs/1412.6071
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
                /// <param name="output_size">The target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH</param>
                /// <param name="output_ratio">If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)</param>
                /// <returns></returns>
                public static (Tensor Values, Tensor Indices) fractional_max_pool3d_with_indices(Tensor input, long[] kernel_size, long[] output_size = null, double[] output_ratio = null)
                {
                    if (kernel_size == null || kernel_size.Length != 3)
                        throw new ArgumentException("Kernel size must contain three elements.");
                    if (output_size != null && output_size.Length != 3)
                        throw new ArgumentException("output_size must contain three elements.");
                    if (output_ratio != null && output_ratio.Length != 3)
                        throw new ArgumentException("output_ratio must contain three elements.");
                    if (output_size == null && output_ratio == null)
                        throw new ArgumentNullException("Only one of output_size and output_ratio may be specified.");
                    if (output_size != null && output_ratio != null)
                        throw new ArgumentNullException("FractionalMaxPool3d requires specifying either an output size, or a pooling ratio.");
                    if (output_ratio != null && input.ndim != 5)
                        // Not sure why this is the case, but there's an exception in the native runtime
                        // unless there's both a batch dimension and a channel dimension.
                        throw new ArgumentException("FractionalMaxPool3d: input tensor must have 5 dimensions: [N, C, D, H, W]");

                    output_size ??= Array.Empty<long>();
                    output_ratio ??= Array.Empty<double>();

                    unsafe {
                        fixed (long* pkernel_size = kernel_size, poutputSize = output_size) {
                            fixed (double* poutputRatio = output_ratio) {
                                var resOutput = THSTensor_fractional_max_pool3d(input.Handle, (IntPtr)pkernel_size, kernel_size.Length, (IntPtr)poutputSize, output_size.Length, (IntPtr)poutputRatio, output_ratio.Length, out var resIndices);
                                if (resOutput == IntPtr.Zero || resIndices == IntPtr.Zero) { torch.CheckForErrors(); }
                                return (new Tensor(resOutput), new Tensor(resIndices));
                            }
                        }
                    }
                }
            }
        }
    }
}
