(ql:quickload '(:vellum-csv :statistical-learning))


(cl:defpackage #:house-price-example
  (:use #:cl #:statistical-learning.aux-package))

(cl:in-package #:house-price-example)

(defparameter *data*
  (handler-bind
      ((vellum-csv:could-not-parse #'vellum-csv:place-null))
    (vellum:copy-from :csv (~>> (asdf:system-source-directory :statistical-learning)
                                (merge-pathnames "examples/house-price.csv"))
                      :columns '((:name id :type fixnum)
                                 (:name ms-sub-class :type fixnum)
                                 (:name ms-zoning :type string)
                                 (:name lot-frontage :type fixnum)
                                 (:name lot-area :type fixnum)
                                 (:name street :type string)
                                 (:name alley :type string)
                                 (:name lot-shape :type string)
                                 (:name land-contour :type string)
                                 (:name utilities :type string)
                                 (:name lot-config :type string)
                                 (:name land-slope :type string)
                                 (:name neighborhood :type string)
                                 (:name condition1 :type string)
                                 (:name condition2 :type string)
                                 (:name bldg-type :type string)
                                 (:name house-style :type string)
                                 (:name overall-qual :type fixnum)
                                 (:name overall-cond :type fixnum)
                                 (:name year-built :type fixnum)
                                 (:name year-remod-add :type fixnum)
                                 (:name roof-style :type string)
                                 (:name roof-matl :type string)
                                 (:name exterior1st :type string)
                                 (:name exterior2nd :type string)
                                 (:name mas-vnr-type :type string)
                                 (:name mas-vnr-area :type fixnum)
                                 (:name exter-qual :type string)
                                 (:name exter-cond :type string)
                                 (:name foundation :type string)
                                 (:name bsmt-qual :type string)
                                 (:name bsmt-cond :type string)
                                 (:name bsmt-exposure :type string)
                                 (:name bsmt-fin-type1 :type string)
                                 (:name bsmt-fin-sf1 :type fixnum)
                                 (:name bsmt-fin-type2 :type string)
                                 (:name bsmt-fin-sf2 :type fixnum)
                                 (:name bsmt-unf-sf :type fixnum)
                                 (:name total-bsmt-sf :type fixnum)
                                 (:name heating :type string)
                                 (:name heating-qc :type string)
                                 (:name central-air :type boolean)
                                 (:name electrical :type string)
                                 (:name 1st-flr-sf :type fixnum)
                                 (:name 2nd-flr-sf :type fixnum)
                                 (:name low-qual-fin-sf :type fixnum)
                                 (:name gr-liv-area :type fixnum)
                                 (:name bsmt-full-bath :type fixnum)
                                 (:name bsmt-half-bath :type fixnum)
                                 (:name full-bath :type fixnum)
                                 (:name half-bath :type fixnum)
                                 (:name bedroom-abv-gr :type fixnum)
                                 (:name kitchen-abv-gr :type fixnum)
                                 (:name kitchen-qual :type string)
                                 (:name tot-rms-abv-grd :type string)
                                 (:name functional :type string)
                                 (:name fireplaces :type string)
                                 (:name fireplace-qu :type string)
                                 (:name garage-type :type string)
                                 (:name garage-yr-blt :type fixnum)
                                 (:name garage-finish :type string)
                                 (:name garage-cars :type fixnum)
                                 (:name garage-area :type fixnum)
                                 (:name garage-qual :type string)
                                 (:name garage-cond :type string)
                                 (:name paved-drive :type boolean)
                                 (:name wood-deck-sf :type fixnum)
                                 (:name open-porch-sf :type fixnum)
                                 (:name enclosed-porch :type fixnum)
                                 (:name 3-ssn-porch :type fixnum)
                                 (:name screen-porch :type fixnum)
                                 (:name pool-area :type fixnum)
                                 (:name pool-qc :type string)
                                 (:name fence :type string)
                                 (:name misc-feature :type string)
                                 (:name misc-val :type fixnum)
                                 (:name mo-sold :type fixnum)
                                 (:name yr-sold :type fixnum)
                                 (:name sale-type :type string)
                                 (:name sale-condition :type string)
                                 (:name sale-price :type fixnum))
                      :null-strings '("NA"))))

(defparameter *train-data-frame*
  (vellum:select *data*
    :columns (vellum:s (vellum:between :from 1 :to 80))))

(defparameter *target-data-frame*
  (vellum:select *data*
    :columns '(sale-price)))

(defparameter *train-data*
  (~> *train-data-frame*
      sl.encode:make-encoders
      (sl.encode:encode *train-data-frame*)))

(defparameter *target-data*
  (~> *target-data-frame*
      sl.encode:make-encoders
      (sl.encode:encode *target-data-frame*)))

(defparameter *training-parameters*
  (make 'statistical-learning.dt:regression
        :optimized-function (sl.opt:squared-error)
        :maximal-depth 7
        :minimal-difference 0.0000001d0
        :minimal-size 5
        :splitter (sl.common:lift (make-instance 'sl.tp:random-attribute-splitter)
                                  'sl.tp:random-splitter
                                  :trials-count 500)
        :parallel nil))

(defparameter *forest-parameters*
  (make 'statistical-learning.ensemble:random-forest
        :trees-count 5
        :parallel nil
        :tree-batch-size 5
        :tree-attributes-count 20
        :data-points-sampler (make-instance 'sl.ensemble:weights-based-data-points-sampler
                                            :sampling-rate 0.4)
        :tree-parameters *training-parameters*))

(defun refine (model train-data target-data)
  (~> model
      (sl.ensemble:refine-trees
       (make-instance 'statistical-learning.gradient-descent-refine:parameters
                      :epochs 30
                      :sample-size 750
                      :shrinkage 1.0)
       _
       train-data
       target-data)))

(defparameter *forest*
  (refine
   (sl.mp:make-supervised-model *forest-parameters*
                                *train-data*
                                *target-data*
                                :parallel nil)
   *train-data*
   *target-data*))

(cl-user::cinspect *forest*)
