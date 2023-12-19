from library.packages import *

class RerankFunction():
    
    def __init__(
        self, 
        user: str, 
        products: List[str], 
        model: implicit.cpu.als.AlternatingLeastSquares
    ) -> None:
        """
        Initializes an instance of my class RerankFunction.
        Args:
            user (str): initial userid of the user,
            products (list(str)): initial list of products numbers,
            model (model): ALS input model
        """
        self.user = user
        self.products = products
        self.model = model
        self.usersVectors = model.users_factors 
        self.itemsVectors = model.items_factors
        
    def __str__(
        self
    ) -> str:
        """
        Returns a string representation of the instance.
        Returns:
            str: Provide information of the instance: nº of users and list of products.
        """
        state = '{} \n\nUser: {} \nList of products: {}'.format(self.model, self.user, self.products)
        return state
        
    def select_vector_user(
        self
    ) -> np.array:
        """
        This method select the latent vector of a specific user
        Returns:
            user_vect (ndarray): values of the latent vector user
        """
        user_vect_row = self.usersVectors.loc[self.user]
        user_vect = np.array(user_vect_row, dtype=float)
        return user_vect
    
    def select_vectors_products(
        self
    ) -> np.array:
        """
        This method select the latent vectors of the list of items
        Returns:
            product_vect (ndarray): values of the latent vectors items
        """
        product_vect_rows = [self.itemsVectors.loc[product] for product in self.products]
        product_vect = np.vstack(product_vect_rows)
        return product_vect
        
    def norm_vect(
        self
    ) -> float:
        """
        This method compute the norm of the latent vectors for the user and all the products
        Returns:
            user_norm (float): user latent vector norm
            product_norm (list[float]): items latent vectors norm
        """
        user_norm = np.linalg.norm(self.select_vector_user())
        product_norm = np.linalg.norm(self.select_vectors_products())
        return user_norm, product_norm
    
    def calculate_scores(
        self,
        scores_func: callable
    ) -> List[float]:
        """
        This method compute the interaction between user and items latent vectors
        Args:
            scores_func (function): user latent vector norm
        Returns:
            scores (list[float]): list of scores from the interaction between latent vectors
        """
        user_vect = self.select_vector_user()
        product_vect = self.select_vectors_products()
        scores = scores_func(user_vect, product_vect)
        return scores
    
    def predict_cosine(
        self
    ) -> Tuple[str, float]:
        """
        This method compute the cosine similitud between the user and item latent vectors
        Returns:
            prediction (List[str]): list of items sorted by cosine similitud
            scores (List[float]: list of scores from cosine similitud
        """
        if len(self.select_vector_user()) == 0 or len(self.select_vectors_products()) == 0:
            return []
        else:
            norm_u, norm_p = self.norm_vect()
            scores = self.calculate_scores(lambda u,p: np.dot(u, p.T)/(norm_u*norm_p))
            prediction = np.array([self.products[idx] for idx in np.argsort(-scores[0])])
            return prediction, scores[0]
        
    def predict_dot_product(
        self
    ) -> Tuple[str, float]:
        """
        This method compute the dot product between the user and item latent vectors
        Returns:
            prediction (List[str]): list of items sorted by cosine similitud
            scores (List[float]: list of scores from cosine similitud
        """
        if len(self.select_vector_user()) == 0 or len(self.select_vectors_products()) == 0:
            return []
        else:
            scores = self.calculate_scores(lambda u,p: np.dot(u, p.T))
            prediction = np.array([self.products[idx] for idx in np.argsort(-scores[0])])
            return prediction, scores[0]
    
    def predict_euclidean_distance(
        self
    ) -> Tuple[str, float]:
        """
        This method compute the euclidean distance between the user and item latent vectors
        Returns:
            prediction (List[str]): list of items sorted by cosine similitud
            scores (List[float]: list of scores from cosine similitud
        """
        if len(self.select_vector_user()) == 0 or len(self.select_vectors_products()) == 0:
            return []
        else:
            scores = self.calculate_scores(lambda u,p: np.linalg.norm(u - p, axis=1))
            prediction = np.array([self.products[idx] for idx in np.argsort(scores)])
            return prediction, scores[0]