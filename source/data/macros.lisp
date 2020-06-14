(cl:in-package #:statistical-learning.data)


(defmacro bind-data-matrix-dimensions (bindings &body body)
  (let ((variables (flatten (mapcar (curry #'take 2) bindings))))
    `(bind ,(mapcar (lambda (binding)
                      (list (take 2 binding)
                            `(data-matrix-dimensions ,(third binding))))
                    bindings)
       (declare (type fixnum ,@variables)
                (ignorable ,@variables))
       ,@body)))


(defmacro check-data-points (&rest data)
  `(progn
     ,@(iterate
         (for d in data)
         (collect `(check-type ,d data-matrix)))
     ,(when (> (length data) 1)
        `(unless (cl-ds.utils:homogenousp (list ,@data)
                                          :key #'data-points-count)
           (error 'cl-ds:incompatible-arguments
                  :parameters '(,@data)
                  :values (list ,@data)
                  :format-control "All data-matrixes are supposed to have equal number of data-points.")))))
