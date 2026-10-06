{
  description = "Serron: mathematical morphology operators for PyTorch, built for the Hugging Face Kernel Hub";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
  };

  outputs =
    { self, kernel-builder, ... }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
