import torch
import math
import triton
import triton.language as tl
import torch.nn.functional as F
import sys
import time


dtype = torch.float32
sm_scale = 1.3
causal = False

@triton.jit
def _bwd_kernel_one_col_block(Q, K, V, P, sm_scale, qk_scale,  #
                              Out, DO,  #
                              DQ, DK, DV,  #
                              D,  #
                              Q_block_ptr, K_block_ptr, V_block_ptr, P_block_ptr, #
                              DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
                              stride_dqa, stride_qm, stride_qk,  #
                              stride_kn, stride_kk,  #
                              stride_vn, stride_vk,  #
                              N_CTX,  #
                              start_n, num_block,  #
                              BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                              BLOCK_N: tl.constexpr,  #
                              ):
    lo = 0
    Q_block_ptr = tl.advance(Q_block_ptr, (lo, 0))
    K_block_ptr = tl.advance(K_block_ptr, (start_n * BLOCK_M, 0))
    V_block_ptr = tl.advance(V_block_ptr, (start_n * BLOCK_M, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo, 0))
    DQ_block_ptr = tl.advance(DQ_block_ptr, (lo, 0))
    DK_block_ptr = tl.advance(DK_block_ptr, (start_n * BLOCK_M, 0))
    DV_block_ptr = tl.advance(DV_block_ptr, (start_n * BLOCK_M, 0))
    
    P_block_ptr = tl.advance(P_block_ptr,(0,start_n*BLOCK_M))
    
    offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_N)
    D_ptrs = D 
    dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    
    for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        q = tl.load(Q_block_ptr)
        # Loading Attention scores on-chip
        p = tl.load(P_block_ptr)
        # compute dv
        do = tl.load(DO_block_ptr)
        dv += tl.dot(tl.trans(p), do)
        # compute dp = dot(v, do)
        Di = tl.load(D_ptrs + offs_m_curr)
        # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp = tl.dot(do, tl.trans(v))
        # compute ds = p * (dp - delta[:, None])
        ds = (p * (dp - Di[:, None]) * sm_scale)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q)
        # compute dq
        
        dq = tl.load(DQ_block_ptr)
        dq += tl.dot(ds, k)
        tl.store(DQ_block_ptr, dq)

        # increment pointers
        DQ_block_ptr = tl.advance(DQ_block_ptr, (BLOCK_M, 0))
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
        P_block_ptr = tl.advance(P_block_ptr,(BLOCK_M,0))

    # write-back
    tl.store(DV_block_ptr, dv.to(V.dtype.element_ty))
    tl.store(DK_block_ptr, dk.to(K.dtype.element_ty))
    
@triton.jit
def _bwd_kernel(Q, K, V, P, sm_scale,  #
                Out, DO,  #
                DQ, DK, DV,  #
                D,  #
                stride_dqa, stride_qm, stride_qk,  #
                stride_kn, stride_kk,  #
                stride_vn, stride_vk,  #
                stride_pm, stride_pn,  #
                N_CTX,  #
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                BLOCK_N: tl.constexpr,  #
                ):
    qk_scale = sm_scale * 1.44269504

    
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    P_block_ptr = tl.make_block_ptr(
        base=P,
        shape=(N_CTX, N_CTX),
        strides=(stride_pm, stride_pn),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_M),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    DQ_block_ptr = tl.make_block_ptr(
            base=DQ,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
    DK_block_ptr = tl.make_block_ptr(
        base=DK,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(Q, K, V, P, sm_scale, qk_scale, Out, DO,  #
                                      DQ, DK, DV,  #
                                      D,  #
                                      Q_block_ptr, K_block_ptr, V_block_ptr, P_block_ptr, #
                                      DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
                                      stride_dqa, stride_qm, stride_qk,  #
                                      stride_kn, stride_kk,  #
                                      stride_vn, stride_vk,  #
                                      N_CTX,  #
                                      start_n, num_block_n,  #
                                      BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,  #
                                      BLOCK_N=BLOCK_N,  #
                                      )

    
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, P, O, #
              stride_qm, stride_qk,  #
              stride_kn, stride_kk,  #
              stride_vn, stride_vk,  #
              stride_pm, stride_pn,
              stride_om, stride_on,#
              N_CTX: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              ):
    pid = tl.program_id(0)

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    P_block_ptr = tl.make_block_ptr(
        base=P,
        shape=(N_CTX, N_CTX),
        strides=(stride_pm, stride_pn),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0))
    
    # acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    Out = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale)
    lo, hi = 0, N_CTX 
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        qk += tl.dot(q, k)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    lo, hi = 0, N_CTX 
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        # acc *= alpha[:, None]
        acc += tl.dot(p, v)
        # -- update m_i and l_i --
        m_i = m_i_new
        # update pointers
        p = p/ l_i[:, None]
        # qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        tl.store(P_block_ptr, p)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        P_block_ptr = tl.advance(P_block_ptr, (0, BLOCK_N))
    # # write back O
    acc = acc / l_i[:, None]
    O_block_ptr = tl.make_block_ptr(
        base=O,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0))
    tl.store(O_block_ptr, acc)
    
@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)

class custom_flash_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        assert Lq == Lk and (Lk == Lv or v.dtype == torch.float8_e5m2)
        p = torch.randn((q.shape[0], q.shape[0]), dtype=dtype, device="cuda", requires_grad=True)
        out = torch.randn((q.shape[0], Lk), dtype=dtype, device="cuda", requires_grad=True)
        BLOCK_M = 64
        BLOCK_N = 64 
        grid = (triton.cdiv(q.shape[0], BLOCK_M), 1)
        # print("grid: ", grid)
        _attn_fwd[grid](
            q, k, v, sm_scale, p, out, #
            q.stride(0), q.stride(1),  #
            k.stride(0), k.stride(1),  #
            v.stride(0), v.stride(1), #
            p.stride(0), p.stride(1),  #
            out.stride(0), out.stride(1),
            N_CTX=q.shape[0],  #
            BLOCK_M=BLOCK_M,  #
            BLOCK_N=BLOCK_N,  #
            BLOCK_DMODEL=Lk  #
        )
        ctx.save_for_backward(q, k, v, out, p)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return out
        
    @staticmethod
    def backward(ctx, do):

        BLOCK = 64
        q, k, v, out, p = ctx.saved_tensors
        seq_len_kv = k.shape[0]
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=q.dtype)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty((q.shape[0]), device=q.device, dtype=torch.float32)
        _bwd_preprocess[(triton.cdiv(q.shape[0], BLOCK), )](
            o,
            do,
            delta,
            BLOCK_M=BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
        )

        _bwd_kernel[(1,)](
            q, k, v, p, ctx.sm_scale,  #
            o, do,  #
            dq, dk, dv,  #
            delta,  #
            o.numel(), q.stride(0), q.stride(1), #
            k.stride(0), k.stride(1),  #
            v.stride(0), v.stride(1),
            p.stride(0), p.stride(1), #
            q.shape[0],  #
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,  #
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,  #
            num_warps=8,  #
            num_stages=2  #
        )

        
        if len(dq.shape) == 5:
            dq = dq.sum(dim=0)
        return dq, dk, dv, None, None, None