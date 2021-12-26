(cl:in-package #:statistical-learning.club-drf)


(defun diversity-measure (tree/labels-a tree/labels-b)
  (bind (((tree-a . labels-a) tree/labels-a)
         ((tree-b . labels-b) tree/labels-b))
    (declare (ignore tree-a tree-b))
    (check-type labels-a sl.data:double-float-data-matrix)
    (check-type labels-b sl.data:double-float-data-matrix)
    (assert (= (sl.data:data-points-count labels-a)
               (sl.data:data-points-count labels-b)))
    (assert (= (sl.data:attributes-count labels-a)
               (sl.data:attributes-count labels-b)))
    (iterate
      (declare (optimize (speed 3) (safety 0))
               (type fixnum data-points-count attributes-count
                     different max-a max-b))
      (with different = 0)
      (with data-points-count = (sl.data:data-points-count labels-a))
      (with attributes-count = (sl.data:attributes-count labels-a))
      (for data-point from 0 below data-points-count)
      (for max-a = (iterate
                     (declare (type fixnum attribute))
                     (for attribute from 0 below attributes-count)
                     (finding attribute maximizing
                              (sl.data:mref labels-a data-point attribute))))
      (for max-b = (iterate
                     (declare (type fixnum attribute))
                     (for attribute from 0 below attributes-count)
                     (finding attribute maximizing
                              (sl.data:mref labels-b data-point attribute))))
      (unless (= max-a max-b) (incf different))
      (finally (return (coerce different 'single-float))))))


(defun obtain-labels (algorithm ensemble train-data)
  (funcall (if (parallel algorithm) #'lparallel:pmap #'cl:map)
           'vector
            (lambda (tree)
              (~>> (sl.ensemble:trees-predict ensemble
                                              (vector tree)
                                              train-data
                                              nil)
                   (cons tree)))
            (sl.ensemble:trees ensemble)))
