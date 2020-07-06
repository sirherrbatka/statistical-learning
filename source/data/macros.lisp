(cl:in-package #:statistical-learning.data)


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


(defmacro dispatch-data-matrix (datums &body body)
  `(cl-ds.utils:cases ,(mapcar (lambda (x)
                                 `(:variant (typep ,x 'universal-data-matrix)
                                            (typep ,x 'double-float-data-matrix)))
                               datums)
     ,@body))


(defmacro bind-data-matrix-dimensions (bindings &body body)
  (let ((variables (flatten (mapcar (curry #'take 2) bindings))))
    `(bind ,(mapcar (lambda (binding)
                      (list (take 2 binding)
                            `(data-matrix-dimensions ,(third binding))))
                    bindings)
       (declare (type fixnum ,@variables)
                (ignorable ,@variables))
       (dispatch-data-matrix (,@(mapcar #'third bindings))
         ,@body))))
