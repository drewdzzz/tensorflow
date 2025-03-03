load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_native_cc_binary")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")

package(
    default_compatible_with = get_compatible_with_cloud(),
    default_visibility = [
        "//tensorflow/compiler/mlir/tensorflow:__subpackages__",
        "//tensorflow/core:__subpackages__",
        "//tensorflow/tools/tfg_graph_transforms:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

gentbl_cc_library(
    name = "PassIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "--name",
                "TFGraph",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "PassDetail",
    hdrs = ["pass_detail.h"],
    deps = [
        ":PassIncGen",
        "//tensorflow/core/ir:Dialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "PassRegistration",
    hdrs = ["pass_registration.h"],
    deps = [
        ":PassIncGen",
        "//tensorflow/core/transforms/const_dedupe_hoist:Pass",
        "//tensorflow/core/transforms/drop_unregistered_attribute:OutputShapesPass",
        "//tensorflow/core/transforms/graph_to_func:Pass",
        "//tensorflow/core/transforms/toposort:Pass",
    ],
)

cc_library(
    name = "graph_transform_wrapper",
    srcs = ["graph_transform_wrapper.cc"],
    hdrs = ["graph_transform_wrapper.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core/ir/importexport:export",
        "//tensorflow/core/ir/importexport:import",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "graph_transform_wrapper_test",
    srcs = ["graph_transform_wrapper_test.cc"],
    deps = [
        ":graph_transform_wrapper",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:graph_constructor",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/platform:status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

# Custom `mlir-opt` replacement that links our dialect and passes
tf_native_cc_binary(
    name = "tfg-transforms-opt",
    srcs = ["tfg-transforms-opt.cc"],
    deps = [
        ":PassRegistration",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Transforms",
    ],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":tfg-transforms-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
