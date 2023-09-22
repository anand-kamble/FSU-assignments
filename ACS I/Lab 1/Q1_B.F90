    
      ! ==============================================      
      ! Question 1 B
      ! ==============================================
      PROGRAM Q1_B
      INTEGER          N, NRHS
      PARAMETER        ( N = 2, NRHS = 2 )
      INTEGER          LDA, LDB
      PARAMETER        ( LDA = N, LDB = N )
      INTEGER          INFO
      INTEGER          IPIV( N )
      DOUBLE PRECISION A( LDA, N), B( LDB, NRHS)
      EXTERNAL DGESV
        
      DATA  A/ 4.5, 1.6, 3.1, 1.1 /

      DATA  B/ 19.249, 6.843 , 19.25, 6.84/

      

      WRITE(*,*) ''
      WRITE(*,*) 'Initialized Matrix A:'
      DO i = 1, N
        WRITE(*,*) (A(i, j), j = 1, 2)
      END DO

      WRITE(*,*) ''
      WRITE(*,*) 'Initialized Matrix B:'
      DO i = 1, LDB
        WRITE(*,*) (B(i,j), j = 1, NRHS)
      END DO




      CALL DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
      WRITE(*,*) ''
      WRITE(*,*) 'Solution to Linear System 1 (X):'
      DO i = 1, LDB
        WRITE(*,*) (B(i, j), j= 1, NRHS)
      END DO
        
 
      END