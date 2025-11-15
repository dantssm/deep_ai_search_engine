import uvicorn

if __name__ == "__main__":
    print("\nDeep Research Engine")
    print("=" * 40)
    print("Web Interface: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 40 + "\n")
    
    uvicorn.run(
        "src.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )