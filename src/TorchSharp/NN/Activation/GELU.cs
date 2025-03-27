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
        /// This class is used to represent a GELU module.
        /// </summary>
        public sealed class GELU : ParameterLessModule<Tensor, Tensor>
        {
            internal GELU(string approximate, bool inplace) : base(nameof(GELU))
            {
                this.approximate = approximate;
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.gelu(tensor, approximate, inplace);
            }

            public bool inplace {get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Gaussian Error Linear Units
            /// </summary>
            public static GELU GELU()
            {
                return new GELU(false);
            }

            /// <summary>
            /// Gaussian Error Linear Units
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            public static GELU GELU(bool inplace)
            {
                return new GELU(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Gaussian Error Linear Units
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                public static Tensor gelu(Tensor x, bool inplace, string approximate = "none")
                {
                    return inplace ? x.gelu_(approximate).alias() : x.gelu(approximate);
                }

                /// <summary>
                /// Gaussian Error Linear Units
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <remarks>The defaulting of 'inplace' to 'false' is implemented as an overload to avoid a breaking change.</remarks>
                public static Tensor gelu(Tensor x, string approximate = "none")
                {
                    if (approximate?.Equals("none") == true) {
                        return gelu(x, false);
                    } else if (approximate?.Equals("tanh") == true) {
                        return gelu(x, false, approximate);
                    } else {
                        throw new ArgumentException($"Unknown approximation method: {approximate}");
                    }
                }
            }
        }
    }
}
