Expr fuse_rule()    {
    // function to create fuse rule 
    if(tree->child!=NULL)   {
        delete();   // create 
        combine node together
        if(op=="divide")    {
            // operation is divide
        } 
        conv2d_relu_add();
        conv2d_add();
    } else  {
        fuse_rule();
    }
}
