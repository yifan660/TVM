class BufferFlattener : public StmtExprMutator  {
    public:
        static PrimFunc Flatten(PrimFunc func)  {
            Map<Var, Buffer> preflattened_buffer_map = Merge(func->buffer_map, func->preflattened_buffer_map);
            auto pass = BufferFlattener(func->buffer_map);
            auto writer = func.CopyOnWrite();
            writer->body = pass.VisitStmt();
            writer->preflattened_buffer_map = preflattened_buffer_map;
            writer->buffer_map = pass.updated_extern_buffer_map_;
            return func;
        }
    
    private:
        explicit BufferFlattener(const Map<Var, Buffer>& extern_buffer_map) {
            for(const auto& kv : extern_buffer_map)   {
                updated_extern_buffer_map_.Set(kv.first, GetFlattenedBuffer(kv.second));
            }
        }

        Stmt VisitStmt_(const BlockNode* op) final  {
            ICHECK_EQ(op->match_buffer.size(),0) << "Unexpected MatchBufferRegion found during tir.transform.FlattenBuffer." << "All MatchBufferRegion should be removed in tir.transform.LowerMatchBuffer.";
            Block block = GetRef<Block>(op);
            Array<BufferRegion> alloc_buffers = op->alloc_buffers;
            alloc_buffers.MutateByApply([this](Buffer buf)  { return GetFlattenedBuffer(buf); });
            if(!alloc_buffers.same_as(op->alloc_buffers))   {
                block.CopyOnWrite()->alloc_buffers = alloc_buffers;
            }

            Array<BufferRegion> reads = op->reads;
            reads.MutateByApply([this](BufferRegion region) { return MutateBufferRegion(region); });
            if(!reads.same_as(op->reads))   {
                block.CopyOnWrite()->reads = reads;
            }

            Array<BufferRegion> writes = op->writes;
            writes.MutateByApply([this](BufferRegion region) { return MutateBufferRegion(region); });
            if(!writes.same_as(op->writes))   {
                block.CopyOnWrite()->writes = writes;
            }
        }

        Stmt VisitStmt_(const AllocateNode* op) final {
            Allocate alloc = Downcast<Allocate>(StmtExprMutator::VisitStmt_(op));
            if(alloc->dtype == DataType::Bool())
        }
}

class Allocate : public Stmt    {
    public:
        TVM_DLL Allocate(Var )
};

class AllocateConstNode : public StmtNode   {
    public:
        Var buffer_var;
        Optional<runtime::NDArray> data;
        Optional<Integer> irmod_storage_idx;
        DataType dtype;
        Array<PrimExpr> extents;
        Stmt body;

        void VisitAttrs(AttrVisitor* v)   {
            v->Visit("buffer_var", &);
            v->Visit();
            v->Visit();
            v->Visit();
            v->Visit();
            v->Visit();
            v->Visit();
            v->Visit();
        }
};


