import matplotlib.pyplot as plt
import numpy as np


def matrix_rot(angle=0):
    
    m = [ [  np.cos(angle), -np.sin(angle) ],
          [  np.sin(angle),  np.cos(angle) ] ] 
    
    return np.asarray(m)


def shapeVertices(n=3):
    
    alpha = 2*np.pi/n
    
    # Generating polygon vertices
    v = [] 
    for i in range(n): v.append( [np.cos(i*alpha), np.sin(i*alpha)])
    v.append( v[0] )
    v = np.asarray(v).T

    if  ( n % 2 == 1 ): beta = np.pi/2
    elif( n % 4 == 2 ): beta = 0
    else              : beta = alpha/2

    a = 0.5/np.sin(alpha/2) # Making the side lengths equal to 1
    v = a*np.matmul( matrix_rot(beta), v )

    # Finding triangle vertices
    s  = -np.matmul( matrix_rot(alpha/2), [0,a] )
    v0 = [ [     0,   0  ],
           [  s[0], s[1] ],
           [ -s[0], s[1] ],
           [     0,   0  ] ]

    v0 = np.asarray(v0).T
    
    return (v,v0)


def makePoints(n=3):
    
    N     = n
    theta = 2*np.pi/N
    
    V, _ = shapeVertices(N)

    # Point arrays
    x, y = [], []

    # Generating random points inside the triangle
    pts = 100*N
    M   = 1/np.tan(theta/2)
    for i in range(pts):

        x1 = np.random.uniform(-0.5, 0.5)
        
        Y0 = -M*0.5
        Y1 =  M*x1
        Y2 =  0
        Y3 = -M*(x1-0.5)

        if( x1 <= 0 ): y1 = np.random.uniform( Y0, Y1  ) 
        else         : y1 = np.random.uniform( Y2, Y3 ) - 0.5*M 
            
        x.append(x1)
        y.append(y1)

    # Rotating generated points
    points = np.asarray([x,y]).T
    for i in range( len(points) ): points[i] = np.matmul( matrix_rot( theta*(i % N) ), points[i] )
    
    del x
    del y
    
    # Creating fractal and midpoints: remember, you're doubling the points each time
    for r in range(8):
        mid     = [] 
        for i in range( len(points) ):   

            vertex = V.T[ np.random.randint(0,N) ]

            ax = np.random.randint(-1,1)
            ay = np.random.randint(-1,1)
            
            x = ( vertex[0] + ax*points[i][0] )/2
            y = ( vertex[1] + ay*points[i][1] )/2

            mid.append( [x,y] )   

        mid = np.asarray(mid)

        points = np.concatenate( (points, mid) )
        np.random.shuffle(points)

    return points.T






#############################################################
# Creating math art
#############################################################

images = 10
for I in range(images):

    # Image information
    folder   = 'Fractal Art'
    filename = f'Image#{I+1} (N={N}) Rand-ID: {np.random.randint(1,1000)}.png'
    print(filename[:-3] + '  ', end=' ')
    
    L = 40    
    fig, ax = plt.subplots( figsize=[L,L] )
    ax.axis('off')
    #ax.set_xlim(left=-1, right=1)
    #ax.set_ylim(bottom=-1, top=1)
       
    N = np.random.randint(10, 15) # Minimum value must be 3 
    
    # Generating points
    s1 = makePoints(n= (N+3) )
    s2 = makePoints(n= N**2 )
    s  = np.concatenate( ( np.copy(s1.T), np.copy(s2.T) ) )
    np.random.shuffle(s)
    s  = s.T
    
    xx, yy = s[0], s[1]
    xx, yy = xx/max(xx), yy/max(yy) # Normalizing to deal with smaller numbers
    

    # Definining Functions for x and y
    def Fx(XX, YY): 
        
        a = np.random.uniform(0, 1, 3) 
        
        return YY*np.cos(3*np.sin( np.sin(a[0]*XX**2) - a[1]*np.cos(XX - a[2]*YY) ) + 1.1*XX - 0.6*YY)
        
        #return YY*np.sin( np.cos(XX) ) + np.cos(1+np.abs(XX))
        #return np.cos(5*YY + a[0]*np.pi ) + 5*np.sin(YY**2 + a[1]) + XX**2
        #return ( - np.cos( YY ) ) 
        #return np.sin(XX + a[0]*YY) - np.sin(XX - a[1])*XX**2 
    
    def Fy(XX, YY): 
        
        a = np.random.uniform(-1, 1, 3)
        #return XX*np.cos( np.cos(YY) + 3*np.sin( a[0]*YY**2 + a[1] ) - 0.3*XX + 0.1*YY)
    
        return YY*np.cos( np.cos(YY) + 3*np.sin( a[0]*XX**2 + a[1] ) - 0.3*YY + 0.8*XX)
        
        #return np.cos( a[1]*np.sin(2*XX + 3) + a[0]*YY**2 )
        #return 4*np.cos(YY-XX) + np.sqrt( np.abs(YY) )
        #return 4*np.cos(3*YY)*np.sin( a[0]*XX - 2*YY*np.cos(YY**2) )

        
    x = Fx(xx,yy) 
    y = Fy(xx,yy)
    
    # Normalizing, for maybe better handling of values?
    x /= max(x)
    y /= max(y)

    
    
    # Plotting Parameters
    mk = 1
    k  = int( np.random.uniform(0,0.5)*len(s.T) )
    c1 = ( np.random.uniform(0.5,1), 0, 0 ) #(1, 0, 0)
    c2 = ( np.random.uniform(0,0.3), np.random.uniform(0.2,0.3), np.random.uniform(0,0.4) ) #(0.3, 0.1, 0.7)
    c3 = ( np.random.uniform(0,1), np.random.uniform(0,0.1), np.random.uniform(0,0.5) ) #(0, 0.1, 0.1)

    ax.scatter( x[:k], y[:k], color= c1, s=mk)
    ax.scatter( x[k:], y[k:], color= c2, s=mk )

    
    # Extra plot for color blending and fun :)
    ax.scatter( 13*x[::3] - 11*y[::3], x[::3]*np.cos(y[::3] - x[::3]), color= c2, s=mk )

    
    
    # Saving Images
    fig.savefig( folder + '//' + filename)
    print('- Done')
    

