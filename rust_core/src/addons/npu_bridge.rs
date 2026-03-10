//! NPU Bridge stub — Intel NPU via npu_engine.dll.
//! NPU_VERIFY_ENABLED defaults to false; this type is kept for compilation.
pub struct NpuBridge;

impl NpuBridge {
    pub fn dummy() -> Self { NpuBridge }
}
