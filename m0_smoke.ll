; ModuleID = 'builtin.module'
source_filename = "m0_smoke"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

declare void @llvm.trap()

define ptx_kernel void @vecadd(ptr nonnull align 4 noalias %v0, i64 %v1, ptr nonnull align 4 noalias %v2, i64 %v3, ptr %v4, i64 %v5) #0 {
entry:
  %v6 = alloca {  }, align 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v3, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v5, 1
  br label %bb0
bb0:
  %v13 = phi { ptr, i64 } [ %v8, %entry ]
  %v14 = phi { ptr, i64 } [ %v10, %entry ]
  %v15 = phi { ptr, i64 } [ %v12, %entry ]
  %v16 = call i64 @_RINvNtNtCsaJwsjXWWFRB_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsbLgPAZB1Kta_8m0_smoke(ptr %v6) #0
  br label %bb1
bb1:
  %v17 = icmp eq i64 %v16, 18446744073709551615
  br i1 %v17, label %bb12, label %bb9
bb2:
  %v18 = extractvalue { ptr } %v41, 0
  %v19 = extractvalue { ptr, i64 } %v13, 1
  %v20 = icmp ult i64 %v16, %v19
  br i1 %v20, label %bb3, label %bb16
bb3:
  br label %bb4
bb4:
  %v21 = extractvalue { ptr, i64 } %v13, 0
  %v22 = getelementptr inbounds float, ptr %v21, i64 %v16
  %v23 = load float, ptr %v22, align 4
  %v24 = extractvalue { ptr, i64 } %v14, 1
  %v25 = icmp ult i64 %v16, %v24
  br i1 %v25, label %bb5, label %bb17
bb5:
  br label %bb6
bb6:
  %v26 = extractvalue { ptr, i64 } %v14, 0
  %v27 = getelementptr inbounds float, ptr %v26, i64 %v16
  %v28 = load float, ptr %v27, align 4
  %v29 = fadd contract float %v23, %v28
  store float %v29, ptr %v18, align 4
  br label %bb8
bb7:
  br label %bb8
bb8:
  ret void
bb9:
  %v30 = extractvalue { ptr, i64 } %v15, 1
  %v31 = icmp ult i64 %v16, %v30
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb11, label %bb10
bb10:
  %v33 = extractvalue { ptr, i64 } %v15, 0
  %v34 = getelementptr inbounds float, ptr %v33, i64 %v16
  %v35 = insertvalue { ptr } undef, ptr %v34, 0
  %v36 = extractvalue { ptr } %v35, 0
  br label %bb13
bb11:
  br label %bb12
bb12:
  %v37 = inttoptr i64 0 to ptr
  %v38 = insertvalue { ptr } undef, ptr %v37, 0
  %v39 = extractvalue { ptr } %v38, 0
  br label %bb13
bb13:
  %v40 = phi ptr [ %v36, %bb10 ], [ %v39, %bb12 ]
  %v41 = insertvalue { ptr } undef, ptr %v40, 0
  %v42 = extractvalue { ptr } %v41, 0
  %v43 = ptrtoint ptr %v42 to i64
  %v44 = sub i64 %v43, 0
  %v45 = icmp ule i64 %v44, 0
  %v46 = add i64 %v44, 0
  %v47 = select i1 %v45, i64 %v46, i64 1
  %v48 = icmp eq i64 %v47, 1
  br i1 %v48, label %bb2, label %bb14
bb14:
  %v49 = icmp eq i64 %v47, 0
  br i1 %v49, label %bb7, label %bb15
bb15:
  unreachable
bb16:
  call void @llvm.trap() #0
  unreachable
bb17:
  call void @llvm.trap() #0
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define i64 @_RINvNtNtCsaJwsjXWWFRB_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsbLgPAZB1Kta_8m0_smoke(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb7
bb3:
  %v5 = xor i1 %v20, 1
  br i1 %v5, label %bb5, label %bb4
bb4:
  %v6 = icmp ne i64 %v19, 18446744073709551615
  br label %bb6
bb5:
  br label %bb6
bb6:
  %v7 = phi i1 [ %v6, %bb4 ], [ 0, %bb5 ]
  %v8 = xor i1 %v7, 1
  br i1 %v8, label %bb14, label %bb13
bb7:
  %v9 = zext i32 %v2 to i64
  %v10 = zext i32 %v3 to i64
  %v11 = zext i32 %v4 to i64
  %v12 = icmp eq i64 %v10, 0
  br i1 %v12, label %bb10, label %bb8
bb8:
  %v13 = sub i64 18446744073709551615, %v11
  %v14 = udiv i64 %v13, %v10
  %v15 = icmp ugt i64 %v9, %v14
  %v16 = xor i1 %v15, 1
  br i1 %v16, label %bb11, label %bb9
bb9:
  br label %bb10
bb10:
  br label %bb12
bb11:
  %v17 = mul i64 %v9, %v10
  %v18 = add i64 %v17, %v11
  br label %bb12
bb12:
  %v19 = phi i64 [ 18446744073709551615, %bb10 ], [ %v18, %bb11 ]
  %v20 = call i1 @_RINvNtNtCsaJwsjXWWFRB_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsbLgPAZB1Kta_8m0_smoke(ptr %v1) #0
  br label %bb3
bb13:
  %v21 = icmp eq i64 %v19, 18446744073709551615
  br i1 %v21, label %bb14, label %bb15
bb14:
  br label %bb15
bb15:
  %v22 = phi i64 [ %v19, %bb13 ], [ 18446744073709551615, %bb14 ]
  ret i64 %v22
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

define i1 @_RINvNtNtCsaJwsjXWWFRB_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsbLgPAZB1Kta_8m0_smoke(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = icmp eq i8 0, 1
  %v3 = xor i1 %v2, 1
  br i1 %v3, label %bb2, label %bb1
bb1:
  br label %bb8
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  br label %bb3
bb3:
  %v5 = icmp eq i32 %v4, 1
  br i1 %v5, label %bb4, label %bb5
bb4:
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0
  br label %bb6
bb5:
  br label %bb7
bb6:
  %v7 = icmp eq i32 %v6, 1
  br label %bb7
bb7:
  %v8 = phi i1 [ 0, %bb5 ], [ %v7, %bb6 ]
  br label %bb8
bb8:
  %v9 = phi i1 [ 1, %bb1 ], [ %v8, %bb7 ]
  %v10 = xor i1 %v2, 1
  br i1 %v10, label %bb9, label %bb10
bb9:
  %v11 = icmp eq i8 0, 2
  %v12 = xor i1 %v11, 1
  br i1 %v12, label %bb11, label %bb10
bb10:
  br label %bb17
bb11:
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0
  br label %bb12
bb12:
  %v14 = icmp eq i32 %v13, 1
  br i1 %v14, label %bb13, label %bb14
bb13:
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0
  br label %bb15
bb14:
  br label %bb16
bb15:
  %v16 = icmp eq i32 %v15, 1
  br label %bb16
bb16:
  %v17 = phi i1 [ 0, %bb14 ], [ %v16, %bb15 ]
  br label %bb17
bb17:
  %v18 = phi i1 [ 1, %bb10 ], [ %v17, %bb16 ]
  %v19 = xor i1 %v9, 1
  br i1 %v19, label %bb19, label %bb18
bb18:
  br label %bb20
bb19:
  br label %bb20
bb20:
  %v20 = phi i1 [ %v18, %bb18 ], [ 0, %bb19 ]
  ret i1 %v20
}


@llvm.used = appending global [1 x ptr] [ptr @vecadd], section "llvm.metadata"

attributes #0 = { convergent }
