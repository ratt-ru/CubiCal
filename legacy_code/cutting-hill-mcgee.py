import numpy as np
import pylab as plt
from scipy import optimize
import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering, cuthill_mckee_ordering

class bandwidth_red():
   
   def __init__(self):
       pass  
     
   def obtain_phi(self,M,R,N,e):
       phi_v_t = np.zeros((R.shape[0],R.shape[2]))
       G = np.zeros(M.shape)
       one_M = np.ones((R.shape[0],R.shape[0]))

       for t in xrange(R.shape[2]):
       #for t in xrange(1):
           Mt = M[:,:,t]
           Rt = R[:,:,t] 
           phi_v_t[:,t] = self.update_phi(phi_v_t[:,t],Rt,Mt,N,e)
           G[:,:,t] = np.dot(np.diag(np.exp(-1j*phi_v_t[:,t])),one_M)
           G[:,:,t] = np.dot(G[:,:,t],np.diag(np.exp(1j*phi_v_t[:,t])))  
   
       return phi_v_t,G
       
   def update_phi(self,phi,R,M,N,e):
       phi_v = np.copy(phi)+0
       r = self.generate_r(R)
       #print "r = ",r
       for i in xrange(N):
           old_phi = np.copy(phi_v) 
           J = self.generate_Jac_real_data(old_phi,M)
           #print "J = ",J
           Jr = np.dot(J.T,r)
           #print "Jr = ",Jr 
           for j in xrange(len(phi_v)):
               Mj = (np.absolute(M[j,:]))**2
               Mj[j] = 0
               Mj_s = np.sum(Mj)
               #print "Mj_s",Mj_s
               phi_v[j] = old_phi[j] + Jr[j]/Mj_s
               #del_theta[j] = d_H[j]**(-1)*(Jr[j] - np.sum(del_theta*H_0[j,:]))
           diff = np.sqrt(np.sum((old_phi-phi_v)**2))
           norm_x = np.sqrt(np.sum(old_phi**2))
           if diff <= e*(norm_x+e):
              print "i =", i
              break 
       return phi_v     

   def generate_Jac_real_data(self,phi_v,M):
     
       Nant = M.shape[0]
       Nb = (Nant**2 - Nant)/2 
       
       Jr = np.zeros((Nb,Nant))
       Ji = np.zeros((Nb,Nant))
       p = np.zeros((Nb,),dtype=int)
       q = np.zeros((Nb,),dtype=int)
       
       #generate p and q
       counter = 0
       for k in xrange(Nant):
           for j in xrange(k+1,Nant):
               p[counter] = k
               q[counter] = j
               counter = counter + 1
       
       for k in xrange(Nant):
           for j in xrange(Nb):
	     if (k == p[j]) or (k == q[j]):
	        if (k == p[j]):
		    const = -1
		else:
		    const = 1
		amplitude = np.absolute(M[p[j],q[j]])
		#print "amplitude = ",amplitude
		#print "amplitude2 = ",amplitude**2
		phase = -1*np.angle(M[p[j],q[j]])
		#print "phase = ",phase
		Jr[j,k] = const*amplitude*np.sin(phi_v[p[j]]+phase-phi_v[q[j]])
		Ji[j,k] = const*amplitude*np.cos(phi_v[p[j]]+phase-phi_v[q[j]])
		 
       J = np.vstack((Jr,Ji))
       return J

   def generate_r(self,R):
     
       Nant = R.shape[0]
       Nb = (Nant**2 - Nant)/2 
       
       r = np.zeros((Nb,))
       i = np.zeros((Nb,))
              
       #generate p and q
       counter = 0
       for k in xrange(Nant):
           for j in xrange(k+1,Nant):
               r[counter] = R[k,j].real
               i[counter] = R[k,j].imag
               counter = counter + 1
       ri = np.hstack((r,i))
       return ri

   def generate_x(self,phi_v,M):
     
       Nant = M.shape[0]
       Nb = (Nant**2 - Nant)/2 
       
       r = np.zeros((Nb,))
       i = np.zeros((Nb,))
              
       #generate p and q
       counter = 0
       for k in xrange(Nant):
           for j in xrange(k+1,Nant):
               t = np.exp(-1j*phi_v[k])*M[k,j]*np.exp(1j**phi_v[j])
               r[counter] = t.real
               i[counter] = t.imag
               counter = counter + 1
       ri = np.hstack((r,i))
       return ri

   def generate_e(self,phi_v,M,R):
       r = self.generate_r(R)
       x = self.generate_x(phi_v,M)
       return r-x

   def normal_inverse(self,H,Jr,mu):
       del_theta = np.dot(np.linalg.pinv(H + mu*np.diag(H)),Jr)
       #print "del_theta = ",del_theta
       return del_theta

   def app_inverse(self,H,Jr,mu):
       del_theta = np.diag(H + mu*np.diag(H))**(-1)*Jr
       #print "del_theta = ",del_theta
       return del_theta

   def jacobi_inverse(self,H,Jr,mu,N,e):
       
       H = H + mu*np.diag(H) 

       del_theta = np.zeros(Jr.shape)
       
       one = np.ones(H.shape)
       I = np.eye(len(del_theta))
       one_0 = one-I
       
       H_0 = H*one_0
       d_H = np.diag(H)
       
       for i in xrange(N):
           old_theta = np.copy(del_theta) 
           for j in xrange(len(del_theta)):
               del_theta[j] = d_H[j]**(-1)*(Jr[j] - np.sum(del_theta*H_0[j,:]))
           diff = np.sqrt(np.sum((old_theta-del_theta)**2))
           norm_x = np.sqrt(np.sum(old_theta**2))
           if diff <= e*(norm_x+e):
              print "i =", i
              break
       return del_theta
         
   '''Creates the observed visibilities and the predicted visibilities'''
   def create_R(self,point_sources,u,v,error,sig):
       R = np.zeros(u.shape)
       G = np.diag(np.exp(-1j*error))
       for k in xrange(len(point_sources)):
           l_0 = point_sources[k,1]
           m_0 = point_sources[k,2]
           R = R + point_sources[k,0]*np.exp(-2*np.pi*1j*(u*l_0+v*m_0))
       for t in xrange(R.shape[2]):
           R[:,:,t] = np.dot(G,R[:,:,t])
           R[:,:,t] = np.dot(R[:,:,t],G.conj()) + sig*np.random.randn(u.shape[0],u.shape[1]) + sig*np.random.randn(u.shape[0],u.shape[1])*1j
       return R

   '''Creates very simplistic uv-tracks'''
   def simple_uv_tracks(self,Phi,num):
       lambda_v = 3e8/1.445e9
       length = 144/lambda_v
       l12 = length * Phi[0]
       l23 = length * Phi[1]
       l13 = length * Phi[2]
       t = np.linspace(-0.24,0.24,num=num)

       u = np.zeros((3,3,len(t)))
       v = np.copy(u)

       u[0,1,:] = l12*np.cos(2*np.pi*t)
       v[0,1,:] = l12*np.sin(2*np.pi*t)

       u[1,0,:] = -1*u[0,1,:]
       v[1,0,:] = -1*v[0,1,:]

       u[0,2,:] = l13*np.cos(2*np.pi*t)
       v[0,2,:] = l13*np.sin(2*np.pi*t)#+1000

       u[2,0,:] = -1*u[0,2,:]
       v[2,0,:] = -1*v[0,2,:]

       u[1,2,:] = l23*np.cos(2*np.pi*t)
       v[1,2,:] = l23*np.sin(2*np.pi*t)

       u[2,1,:] = -1*u[1,2,:]
       v[2,1,:] = -1*v[1,2,:]

       return u,v
       
   '''Creates very simplistic uv-tracks'''
   def simple_uv_tracks4(self,Phi,num):
       lambda_v = 3e8/1.445e9
       length = 144/lambda_v
       l12 = length * Phi[0]
       l23 = length * Phi[1]
       l13 = length * Phi[2]
       l14 = length * Phi[3]
       l24 = length * Phi[4]
       l34 = length * Phi[5]
       
       t = np.linspace(-0.24,0.24,num=num)

       u = np.zeros((4,4,len(t)))
       v = np.copy(u)

       u[0,1,:] = l12*np.cos(2*np.pi*t)
       v[0,1,:] = l12*np.sin(2*np.pi*t)

       u[1,0,:] = -1*u[0,1,:]
       v[1,0,:] = -1*v[0,1,:]

       u[0,2,:] = l13*np.cos(2*np.pi*t)
       v[0,2,:] = l13*np.sin(2*np.pi*t)#+1000

       u[2,0,:] = -1*u[0,2,:]
       v[2,0,:] = -1*v[0,2,:]

       u[1,2,:] = l23*np.cos(2*np.pi*t)
       v[1,2,:] = l23*np.sin(2*np.pi*t)

       u[2,1,:] = -1*u[1,2,:]
       v[2,1,:] = -1*v[1,2,:]
       
       u[0,3,:] = l14*np.cos(2*np.pi*t)
       v[0,3,:] = l14*np.sin(2*np.pi*t)

       u[3,0,:] = -1*u[0,3,:]
       v[3,0,:] = -1*v[0,3,:]
       
       u[1,3,:] = l24*np.cos(2*np.pi*t)
       v[1,3,:] = l24*np.sin(2*np.pi*t)

       u[3,1,:] = -1*u[1,3,:]
       v[3,1,:] = -1*v[1,3,:]
       
       u[2,3,:] = l34*np.cos(2*np.pi*t)
       v[2,3,:] = l34*np.sin(2*np.pi*t)

       u[3,2,:] = -1*u[2,3,:]
       v[3,2,:] = -1*v[2,3,:]
       
          

       return u,v    
       
	     
   def generate_moc_Jac(self,Nant):
       Nb = (Nant**2 - Nant)/2 

       p = np.zeros((Nb,),dtype=int)
       q = np.zeros((Nb,),dtype=int)
       J = np.zeros((Nb,Nant))
        
       #generate p and q
       counter = 0
       for k in xrange(Nant):
           for j in xrange(k+1,Nant):
               p[counter] = k
               q[counter] = j
               counter = counter + 1

       #generate Jacobian
       for k in xrange(Nant):
           for j in xrange(Nb):
               if (k == p[j]) or (k == q[j]):
                  J[j,k] = 1
       
       return J

   def generate_moc_Jac3(self,Nant):
       Nb = (Nant**2 - Nant)/2 

       p = np.zeros((Nb,),dtype=int)
       q = np.zeros((Nb,),dtype=int)
       J = np.zeros((Nb,Nant))
        
       #generate p and q
       counter = 0
       for k in xrange(Nant):
           for j in xrange(k+1,Nant):
               p[counter] = k
               q[counter] = j
               counter = counter + 1

       #generate Jacobian
       for k in xrange(Nant):
           for j in xrange(Nb):
               if (k == p[j]) or (k == q[j]):
                  J[j,k] = 1
       J_top = np.hstack((J,J))
       J_bottom = np.hstack((J,J))

       J = np.vstack((J_top,J_bottom))
       
       return J
   
   def generate_moc_Jac4(self,Nant):
       Nb = (Nant**2 - Nant)/2 

       p = np.zeros((Nb,),dtype=int)
       q = np.zeros((Nb,),dtype=int)
       J = np.zeros((Nb,Nant))
        
       #generate p and q
       counter = 0
       for k in xrange(Nant):
           for j in xrange(k+1,Nant):
               p[counter] = k
               q[counter] = j
               counter = counter + 1

       #generate Jacobian
       for k in xrange(Nant):
           for j in xrange(Nb):
               if (k == p[j]) or (k == q[j]):
                  J[j,k] = 1
       #J_top = np.hstack((J,J))
       #J_bottom = np.hstack((J,J))

       J = np.vstack((J,J))
       
       return J

   def generate_moc_Jac2(self,Nant):
       Nb = (Nant**2 - Nant)/2 

       p = np.zeros((Nb,),dtype=int)
       q = np.zeros((Nb,),dtype=int)
       J1 = np.zeros((Nb,Nant))
       J2 = np.zeros((Nb,Nant))
               
       #generate p and q
       counter = 0
       for k in xrange(Nant):
           for j in xrange(k+1,Nant):
               p[counter] = k
               q[counter] = j
               counter = counter + 1

       #generate Jacobian 1
       for k in xrange(Nant):
           for j in xrange(Nb):
               if (k == p[j]):
                  J1[j,k] = 1

       #generate Jacobian 2
       for k in xrange(Nant):
           for j in xrange(Nb):
               if (k == q[j]):
                  J2[j,k] = 1

       J_top = np.hstack((J1,J2))
       J_bottom = np.hstack((J2,J1))

       J = np.vstack((J_top,J_bottom))
       
       return J

   def create_Hess(self,J):
       return np.dot(J.T,J) 

   def obtain_degree(self,M):
       d = np.zeros((M.shape[0],),dtype=int)
       one = np.ones((M.shape[0],),dtype=int)
       for k in xrange(len(d)):
           m = M[:,k]
           d[k] = np.sum(one[m>0])-1
       return d   
     
   def index_lowest_degree(self,degree,r):
       if len(r) == 0:
          return np.argmin(degree)

       all_ind = np.arange(len(r))
      
       not_in_r = np.setdiff1d(all_ind, r, assume_unique=False)

       sub_degree = degree[not_in_r]

       min_ind = np.argmin(sub_degree)

       return not_in_r[min_ind]

   #def index_lowest_degree(self,degree,r):
   #    found = False
   #    deg = np.copy(degree)+0
   #   if len(r) > 0:
   #       one = np.ones(r.shape,dtype=int)
   #    while not found:
   #          lowest_ind = np.argmin(deg)
   #          if len(r) > 0:
   #             cand =  
   #          else:
   #             found = True
   #          if len(deg) == 0:
   #             found = True
   #             lowest_ind = -1
   #    return lowest_ind  

   def add_it_to_r(self,r,index):
       if len(r)==0:
          r = np.append(r,index)
          added = True
       else:
          one = np.ones(r.shape,dtype=int)
          if np.sum(one[r==index]) == 0:
             r = np.append(r,index)
             added = True
          else:
             added = False
       return r,added 

   def get_children(self,index,M,degree,r):
       m = M[:,index]
       #print "m = ",m  
       ind = np.arange(M.shape[0])
       children = ind[m<>0]
       #print "children1 = ",children
       children = np.setdiff1d(children, r, assume_unique=False) 
       #print "children2 = ",children
       deg_children = degree[children]
       deg_indx_sorted = np.argsort(deg_children)
       children = children[deg_indx_sorted]
       #print "children3 = ",children  
       return children    
        
             
   def reverse_cuthill_mckee(self,M):

       r = np.array([],dtype=int)
       q = np.array([],dtype=int)
      
       degree = self.obtain_degree(M)
       #print "degree = ",degree

       counter = 0       

       #while len(r) < M.shape[0]:
       while len(r) < M.shape[0]:
             counter = counter + 1

             #if counter > 4:
             #   break
             lowest_index = self.index_lowest_degree(degree,r)
             #print "lowest_index = ",lowest_index
             r,added = self.add_it_to_r(r,lowest_index)
             #print "r = ",r
             c = self.get_children(lowest_index,M,degree,r)
             #print "c = ",c
             q = np.append(q,c)
             #print "q = ",q
             while len(q)<>0:
                   #print "**********"
                   #print "r_b = ",r+1
                   #print "q_b = ",q+1 
                   #print "**********"
                   index = q[0]
                   q = np.delete(q,0)
                   r,added = self.add_it_to_r(r,index)
                   if added:
                      c = self.get_children(index,M,degree,r)
                      q = np.append(q,c)
                   #print "**********"
                   #print "r_a = ",r+1
                   #print "q_a = ",q+1 
                   #print "**********"
       return r[::-1]  

if __name__ == "__main__":
   point_sources = np.array([(1,0,0),(0.5,(1*np.pi)/180,(0*np.pi)/180)])
   s = bandwidth_red() #s for solver
   u,v = s.simple_uv_tracks4([2,3,5,7,8,9],1000)
   
   # Plot uv coverage...
   plt.plot(u[0,1,:],v[0,1,:],"b")
   plt.hold("on")
   plt.plot(u[1,0,:],v[1,0,:],"r")
   plt.plot(u[0,2,:],v[0,2,:],"b")
   plt.plot(u[2,0,:],v[2,0,:],"r")
   plt.plot(u[1,2,:],v[1,2,:],"b")
   plt.plot(u[2,1,:],v[2,1,:],"r")
   plt.show()

   M = s.create_R(point_sources,u,v,np.array([0,0,0,0]),0)

   R = s.create_R(point_sources,u,v,np.array([3,1.1,5,2]),0.01)

   phi,G = s.obtain_phi(M,R,200,1e-10)

   #print phi[0,:]
 
   #plt.plot(phi[0,:])
   #plt.hold('on')
   #plt.plot(phi[1,:])
   #plt.plot(phi[2,:])
   #plt.show()

   plt.plot(np.absolute(R[0,1,:]),"b")
   plt.hold('on')
   plt.plot(np.absolute(M[0,1,:]),"r")
   #plt.plot((G[0,1,:]**(-1)*R[0,1,:]).real,"g")
   plt.show()
   

  
  
   #r = s.generate_r(R[:,:,0])
   #x = s.generate_x(np.array([1,1,1,1]),M[:,:,0])
   #e = r-x
   #print "r = ",r
   #print "x = ",x
   #print "e = ",e
   #print "M = ",M

   
   
   #J = s.generate_Jac_real_data(np.array([1,1,1,1]),M[:,:,0])
   #print "J = ",J
   #plt.imshow(J,interpolation='nearest')
   #plt.show()
   #H = s.create_Hess(J)
   #print "H = ",H
   #plt.imshow(H,interpolation='nearest')
   #plt.show()
   
   #for j in xrange (H.shape[0]):
   #    print "H[0,0] = ",H[j,j]
   #    print "H[:,0] = ",np.sum(np.absolute(H[j,:]))-H[j,j]
   """
   Jr = np.dot(J.T,r)
   Jx = np.dot(J.T,x)
   Je = np.dot(J.T,e)

   print "Jr = ",Jr
   print "Je = ",Je
   print "Jx = ",Jx

   dtheta = s.normal_inverse(H,Jr,0.00001)
   dtheta2 = s.app_inverse(H,Jr,0.00001)
   dtheta3 = s.jacobi_inverse(H,Jr,0.00001,100000,1e-8)
   print "dtheta =",dtheta
   print "dtheta2 =",dtheta2
   print "dtheta3 =",dtheta3   
   """
   """
   b = bandwidth_red()
   J = b.generate_moc_Jac4(3)
   #print "J = ",J
   #plt.imshow(J,interpolation='nearest')
   #plt.show()
   H = b.create_Hess(J)
   print "H = ",H
   #H[H<=2]=0
   plt.imshow(H,interpolation='nearest')
   plt.show()

   T = np.copy(H)
   A = np.copy(H)
   #T = np.array(([1,0,0,0,1,0,0,0],[0,1,1,0,0,1,0,1],[0,1,1,0,1,0,0,0],[0,0,0,1,0,0,1,0],[1,0,1,0,1,0,0,0],[0,1,0,0,0,1,0,1],[0,0,0,1,0,0,1,0],[0,1,0,0,0,1,0,1]))

   #print "T = ",T
   
   r = b.reverse_cuthill_mckee(T)
   print "final r = ",r+1 
   P = np.zeros(T.shape,dtype=int)
   for k in xrange(len(r)):
       P[k,r[k]] = 1
   print "P = ",P
   T = np.dot(P,T)
   T = np.dot(T,P.T)
   print "T_new = ",T
   plt.imshow(T,interpolation='nearest')
   plt.show()

   G = nx.Graph(A)
   #rcm = list(reverse_cuthill_mckee_ordering(G))
   rcm = list(reverse_cuthill_mckee_ordering(G))
   print "rcm = ",rcm
   A1 = A[rcm, :][:, rcm]
   plt.imshow(A1,interpolation='nearest')
   plt.show()
   """


