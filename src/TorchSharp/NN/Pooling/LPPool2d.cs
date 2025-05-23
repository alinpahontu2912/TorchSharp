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
        /// This class is used to represent a LPPool2D module.
        /// </summary>
        public sealed class LPPool2d : ParameterLessModule<Tensor, Tensor>
        {
            internal LPPool2d(double norm_type, long[] kernel_size, long[] stride = null, bool ceil_mode = false) : base(nameof(LPPool2d))
            {
                this.norm_type = norm_type;
                this.kernel_size = kernel_size;
                this.stride = stride;
                this.ceil_mode = ceil_mode;
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.lp_pool2d(input, norm_type, kernel_size, stride, ceil_mode);
            }

            public double norm_type { get; set; }
            public long[] kernel_size { get; set; }
            public long[] stride { get; set; }
            public bool ceil_mode { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies a 2D power-average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="norm_type">The LP norm (exponent)</param>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="stride">The stride of the window. Default value is kernel_size</param>
            /// <param name="ceil_mode">Use ceil instead of floor to compute the output shape</param>
            /// <returns></returns>
            public static LPPool2d LPPool2d(double norm_type, long[] kernel_size, long[] stride = null, bool ceil_mode = false)
            {
                return new LPPool2d(norm_type, kernel_size, stride, ceil_mode);
            }

            /// <summary>
            /// Applies a 2D power-average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="norm_type">The LP norm (exponent)</param>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="stride">The stride of the window.</param>
            /// <param name="ceil_mode">Use ceil instead of floor to compute the output shape</param>
            /// <returns></returns>
            public static LPPool2d LPPool2d(double norm_type, long kernel_size, long? stride = null, bool ceil_mode = false)
            {
                return new LPPool2d(norm_type, new[] { kernel_size, kernel_size }, stride.HasValue ? new[] { stride.Value, stride.Value } : null, ceil_mode);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 2D power-average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="norm_type">The LP norm (exponent)</param>
                /// <param name="kernel_size">The size of the window</param>
                /// <param name="stride">The stride of the window. Default value is kernel_size</param>
                /// <param name="ceil_mode">Use ceil instead of floor to compute the output shape</param>
                /// <returns></returns>
                public static Tensor lp_pool2d(Tensor input, double norm_type, long[] kernel_size, long[] stride = null, bool ceil_mode = false)
                {
                    stride ??= Array.Empty<long>();

                    unsafe {
                        fixed (long* pkernel_size = kernel_size, pstrides = stride) {
                            var res = THSTensor_lp_pool2d(input.Handle, norm_type, (IntPtr)pkernel_size, kernel_size.Length, (IntPtr)pstrides, stride.Length, ceil_mode);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                /// <summary>
                /// Applies a 2D power-average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="norm_type">The LP norm (exponent)</param>
                /// <param name="kernel_size">The size of the window</param>
                /// <param name="stride">The stride of the window.</param>
                /// <param name="ceil_mode">Use ceil instead of floor to compute the output shape</param>
                /// <returns></returns>
                public static Tensor lp_pool2d(Tensor input, double norm_type, long kernel_size, long? stride = null, bool ceil_mode = false)
                {
                    return lp_pool2d(input, norm_type, new[] { kernel_size, kernel_size }, stride.HasValue ? new[] { stride.Value, stride.Value } : null, ceil_mode);
                }
            }
        }
    }
}
