from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def serve_map(request):
    """
    Example route for serving map details with frontend.

    Returns:
        To be implemented
    """
    return Response({"message": "Placeholder: Map data is sent using this route."})

@api_view(['GET'])
def make_prediction(request):
    """
    Example route for making prediction.

    Returns:
        To be implemented
    """
    return Response({"message": "Placeholder: Prediction workflow is executed using this route."})
