// build.rs — Compile CUDA kernels to PTX when gpu_cuda feature is enabled.

fn main() {
    #[cfg(feature = "gpu_cuda")]
    {
        println!("cargo:rerun-if-changed=cuda/quant_kernels.cu");

        let out_dir = std::env::var("OUT_DIR").unwrap();
        let ptx_path = format!("{}/quant_kernels.ptx", out_dir);

        // Prefer CUDA 13.1 nvcc, fall back to whichever nvcc is on PATH
        let nvcc = if std::path::Path::new(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe",
        )
        .exists()
        {
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe".to_string()
        } else {
            "nvcc".to_string()
        };

        // Find MSVC cl.exe — nvcc needs it as host compiler on Windows
        let cl_dir = find_msvc_cl_dir();

        let mut cmd = std::process::Command::new(&nvcc);
        cmd.args([
            "-ptx",
            "-arch=sm_120", // RTX 5060 Ti Blackwell
            "--use_fast_math",
            "-o",
            &ptx_path,
            "cuda/quant_kernels.cu",
        ]);

        // Prepend cl.exe directory to PATH so nvcc can find it
        if let Some(cl_path) = &cl_dir {
            let current_path = std::env::var("PATH").unwrap_or_default();
            cmd.env("PATH", format!("{};{}", cl_path, current_path));
            println!("cargo:warning=nvcc using cl.exe from {}", cl_path);
        }

        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("nvcc not found at '{}': {}", nvcc, e));

        assert!(status.success(), "nvcc failed to compile CUDA kernels");

        println!("cargo:rustc-env=QUANT_KERNELS_PTX={}", ptx_path);
    }
}

#[cfg(feature = "gpu_cuda")]
fn find_msvc_cl_dir() -> Option<String> {
    // Search common VS install paths for x64 cl.exe
    // Prefer VS 2022/2019 (CUDA 13.1 only supports these) over VS 18 Insiders
    let vs_roots = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC",
    ];

    for root in &vs_roots {
        let root_path = std::path::Path::new(root);
        if !root_path.exists() {
            continue;
        }
        // Find the latest version directory
        if let Ok(entries) = std::fs::read_dir(root_path) {
            let mut versions: Vec<String> = entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().into_string().ok())
                .collect();
            versions.sort();
            versions.reverse();

            for ver in versions {
                let cl = std::path::Path::new(root)
                    .join(&ver)
                    .join("bin")
                    .join("Hostx64")
                    .join("x64")
                    .join("cl.exe");
                if cl.exists() {
                    return Some(cl.parent().unwrap().to_string_lossy().into_owned());
                }
            }
        }
    }
    None
}
