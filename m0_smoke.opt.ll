; ModuleID = '/home/stevek/work/ptyrs/m0_smoke.ll'
source_filename = "m0_smoke"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

@llvm.used = appending global [1 x ptr] [ptr @vecadd], section "llvm.metadata"

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #0

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @vecadd(ptr noalias nofree nonnull readonly align 4 captures(none) %v0, i64 %v1, ptr noalias nofree nonnull readonly align 4 captures(none) %v2, i64 %v3, ptr nofree writeonly captures(address_is_null) %v4, i64 %v5) #1 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %v9.i = zext nneg i32 %v2.i to i64
  %v10.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v9.i, %v10.i
  %v11.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v11.i
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #3
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #3
  %v5.i = icmp eq i32 %v4.i2, 1
  %v7.i = icmp eq i32 %v6.i, 1
  %v8.not.not.i = and i1 %v5.i, %v7.i
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #3
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #3
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i3 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i3
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v31 = icmp ult i64 %v22.i, %v5
  %or.cond.not = select i1 %.v18.i, i1 %v31, i1 false
  %v34 = getelementptr inbounds [4 x i8], ptr %v4, i64 %v22.i
  %v451 = icmp ne ptr %v4, null
  %v45 = select i1 %or.cond.not, i1 %v451, i1 false
  br i1 %v45, label %bb2, label %bb8

bb2:                                              ; preds = %entry
  %v20 = icmp ult i64 %v22.i, %v1
  br i1 %v20, label %bb4, label %bb16

bb4:                                              ; preds = %bb2
  %v25 = icmp ult i64 %v22.i, %v3
  br i1 %v25, label %bb6, label %bb17

bb6:                                              ; preds = %bb4
  %v22 = getelementptr inbounds [4 x i8], ptr %v0, i64 %v22.i
  %v23 = load float, ptr %v22, align 4
  %v27 = getelementptr inbounds [4 x i8], ptr %v2, i64 %v22.i
  %v28 = load float, ptr %v27, align 4
  %v29 = fadd contract float %v23, %v28
  store float %v29, ptr %v34, align 4
  br label %bb8

bb8:                                              ; preds = %entry, %bb6
  ret void

bb16:                                             ; preds = %bb2
  tail call void @llvm.trap() #3
  unreachable

bb17:                                             ; preds = %bb4
  tail call void @llvm.trap() #3
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #2

attributes #0 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #1 = { convergent nounwind memory(argmem: readwrite, inaccessiblemem: write) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent }
