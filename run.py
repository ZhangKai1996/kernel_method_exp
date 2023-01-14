
def main():
    """SVC(SMO)"""
    from svc import iris_SVC, image_SVC
    # iris_SVC.run()
    image_SVC.run()

    # """SVR"""
    # from svr import SVR, SVR_optimization
    # SVR.run()
    # SVR_optimization.run()
    #
    # """PCA/KPCA visual"""
    # from dimension_reduction import iris_PCA_visual, image_PCA_visual
    # iris_PCA_visual.run()
    # image_PCA_visual.run()
    #
    # """PCA/KPCA train"""
    # from svc import iris_SVC_pca_lda
    # iris_SVC_pca_lda.run()
    #
    # """LDA/KLDA visual"""
    # from dimension_reduction import iris_LDA_visual, image_LDA_visual
    # iris_LDA_visual.run()
    # image_LDA_visual.run()
    #
    # """LDA/KLDA train"""
    # from svc import image_SVC_pca_lda
    # image_SVC_pca_lda.run()


if __name__ == '__main__':
    main()
