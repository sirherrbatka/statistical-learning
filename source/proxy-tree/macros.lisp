(cl:in-package #:sl.proxy-tree)


(defmacro define-forwarding (&body forms)
  `(progn
     ,@(iterate
         (with !parameters = (gensym))
         (for (name arguments) in forms)
         (collect `(defmethod ,name ((,!parameters proxy-tree)
                                     ,@arguments)
                     (forward-call ,!parameters
                                   #',name
                                   ,@arguments))))))
